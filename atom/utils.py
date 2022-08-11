# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing utility constants, functions and classes.

"""

import logging
import math
import pprint
import sys
from collections import deque
from collections.abc import MutableMapping
from copy import copy
from datetime import datetime
from functools import wraps
from inspect import Parameter, signature
from typing import Protocol, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import sparse
from shap import Explainer
from sklearn.inspection._partial_dependence import (
    _grid_from_X, _partial_dependence_brute,
)
from sklearn.metrics import (
    SCORERS, confusion_matrix, make_scorer, matthews_corrcoef,
)
from sklearn.utils import _print_elapsed_time, _safe_indexing


# Current library version
__version__ = "5.0.0"

# Group of variable types for isinstance
SEQUENCE = (list, tuple, np.ndarray, pd.Series)

# Groups of variable types for type hinting
INT = Union[int, np.integer]
FLOAT = Union[float, np.float]
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

# List of custom metric acronyms for the evaluate method
CUSTOM_METRICS = (
    "cm",
    "tn",
    "fp",
    "fn",
    "tp",
    "lift",
    "fpr",
    "tpr",
    "fnr",
    "tnr",
    "sup",
)

# Acronyms for some common scorers
SCORERS_ACRONYMS = dict(
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


# Functions ======================================================== >>

def flt(item):
    """Utility to reduce sequences with just one item."""
    if isinstance(item, SEQUENCE) and len(item) == 1:
        return item[0]
    else:
        return item


def lst(item):
    """Utility used to make sure an item is iterable."""
    if isinstance(item, (dict, CustomDict, *SEQUENCE)):
        return item
    else:
        return [item]


def it(item):
    """Utility to convert rounded floats to int."""
    try:
        is_equal = int(item) == float(item)
    except ValueError:  # Item may not be numerical
        return item

    return int(item) if is_equal else float(item)


def divide(a, b):
    """Divide two numbers and return 0 if division by zero."""
    return np.divide(a, b) if b != 0 else 0


def merge(X, y):
    """Merge a pd.DataFrame and pd.Series into one dataframe."""
    return X.merge(y.to_frame(), left_index=True, right_index=True)


def variable_return(X, y):
    """Return one or two arguments depending on which is None."""
    if y is None:
        return X
    elif X is None:
        return y
    else:
        return X, y


def is_sparse(df):
    """Check if the dataframe contains any sparse columns."""
    return any(pd.api.types.is_sparse(df[col]) for col in df)


def check_goal(goal, method, task):
    """Raise an error if the goal is invalid."""
    if not task.startswith(goal):
        raise PermissionError(
            f"The {method} method is only available for {task} tasks!"
        )


def check_binary_task(task, method):
    """Raise an error if the task is invalid."""
    if not task.startswith("bin"):
        raise PermissionError(
            f"The {method} method is only available for binary classification tasks!"
        )


def check_predict_proba(models, method):
    """Raise an error if a model doesn't have a predict_proba method."""
    for m in [m for m in models if m.name != "Vote"]:
        if not hasattr(m.estimator, "predict_proba"):
            raise AttributeError(
                f"The {method} method is only available for "
                f"models with a predict_proba method, got {m.name}."
            )


def check_scaling(X):
    """Check if the data is scaled to mean=0 and std=1."""
    mean = X.mean(numeric_only=True).mean()
    std = X.std(numeric_only=True).mean()
    return True if mean < 0.05 and 0.9 < std < 1.1 else False


def get_corpus(X):
    """Get text column from dataframe."""
    try:
        return next(col for col in X if col.lower() == "corpus")
    except StopIteration:
        raise ValueError("The provided dataset does not contain a text corpus!")


def get_pl_name(name, steps, counter=1):
    """Add a counter to a pipeline name if already in steps."""
    og_name = name
    while name.lower() in [elem[0] for elem in steps]:
        counter += 1
        name = og_name + str(counter)

    return name.lower()


def get_best_score(item, metric=0):
    """Returns the bootstrap or test score of a model.

    Parameters
    ----------
    item: model or pd.Series
        Model instance or row from the results dataframe.

    metric: int, default=0
        Index of the metric to use.

    """
    if getattr(item, "mean_bootstrap", None):
        return lst(item.mean_bootstrap)[metric]
    else:
        return lst(item.metric_test)[metric]


def time_to_str(t_init):
    """Convert time integer to string.

    Convert a time duration to a string of format 00h:00m:00s
    or 1.000s if under 1 min.

    Parameters
    ----------
    t_init: datetime
        Time to convert (in seconds).

    Returns
    -------
    str
        Time representation.

    """
    t = datetime.now() - t_init
    h = t.seconds // 3600
    m = t.seconds % 3600 // 60
    s = t.seconds % 3600 % 60 + t.microseconds / 1e6
    if not h and not m:  # Only seconds
        return f"{s:.3f}s"
    elif not h:  # Also minutes
        return f"{m}m:{s:02.0f}s"
    else:  # Also hours
        return f"{h}h:{m:02.0f}m:{s:02.0f}s"


def to_df(data, index=None, columns=None, dtypes=None):
    """Convert a dataset to pd.Dataframe.

    Parameters
    ----------
    data: list, tuple, dict, np.array, sp.matrix, pd.DataFrame or None
        Dataset to convert to a dataframe.  If None or already a
        dataframe, return unchanged.

    index: sequence or pd.Index
        Values for the dataframe's index.

    columns: sequence or None, default=None
        Name of the columns. Use None for automatic naming.

    dtypes: str, dict, dtype or None, default=None
        Data types for the output columns. If None, the types are
        inferred from the data.

    Returns
    -------
    pd.DataFrame or None
        Transformed dataframe.

    """
    # Get number of columns (list/tuple have no shape and sp.matrix has no index)
    n_cols = lambda data: data.shape[1] if hasattr(data, "shape") else len(data[0])

    if not isinstance(data, pd.DataFrame) and data is not None:
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


def to_series(data, index=None, name="target", dtype=None):
    """Convert a column to pd.Series.

    Parameters
    ----------
    data: sequence or None
        Data to convert. If None, return unchanged.

    index: sequence or Index, default=None
        Values for the indices.

    name: string, default="target"
        Name of the target column.

    dtype: str, np.dtype or None, default=None
        Data type for the output series. If None, the type is
        inferred from the data.

    Returns
    -------
    pd.Series or None
        Transformed series.

    """
    if data is not None and not isinstance(data, pd.Series):
        data = pd.Series(data, index=index, name=name, dtype=dtype)

    return data


def prepare_logger(logger, class_name):
    """Prepare logging file.

    Parameters
    ----------
    logger: str, Logger or None
        - If None: Doesn't save a logging file.
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
            current = datetime.now().strftime("%d%b%y_%Hh%Mm%Ss")
            logger = logger.replace("auto", class_name + "_" + current)

        # Define file handler and set formatter
        file_handler = logging.FileHandler(logger)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)

        # Define logger
        logger = logging.getLogger(class_name + "_logger")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if logger.hasHandlers():  # Remove existing handlers
            logger.handlers.clear()
        logger.addHandler(file_handler)  # Add file handler to logger

    return logger


def check_is_fitted(estimator, exception=True, attributes=None):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (not None or empty). Otherwise it raises a
    NotFittedError. Extension on sklearn's function that accounts
    for empty dataframes and series and returns a boolean.

    Parameters
    ----------
    estimator: class
        Class instance for which the check is performed.

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

    def check_attr(attr):
        """Return empty pandas or None/empty sequence."""
        if attr and isinstance(getattr(estimator, attr), (pd.DataFrame, pd.Series)):
            return getattr(estimator, attr).empty
        else:
            return not getattr(estimator, attr)

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


def create_acronym(fullname):
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


def get_custom_scorer(
    metric,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
):
    """Get a scorer from a str, func or scorer.

    Scorers used by ATOM have a name attribute.

    Parameters
    ----------
    metric: str, func or scorer
        Name, metric or scorer to get ATOM's scorer from.

    greater_is_better: bool, default=True
        whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower is
        better. Is ignored if the metric is a string or a scorer.

    needs_proba: bool, default=False
        Whether the metric function requires probability estimates of
        a classifier. Is ignored if the metric is a string or a scorer.

    needs_threshold: bool, default=False
        Whether the metric function takes a continuous decision
        certainty. Is ignored if the metric is a string or a scorer.

    Returns
    -------
    scorer
        Custom sklearn scorer with name attribute.

    """
    # Copies are needed to not alter SCORERS
    if isinstance(metric, str):
        metric = metric.lower()

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

        if metric in SCORERS:
            scorer = copy(SCORERS[metric])
            scorer.name = metric
        elif metric in SCORERS_ACRONYMS:
            scorer = copy(SCORERS[SCORERS_ACRONYMS[metric]])
            scorer.name = SCORERS_ACRONYMS[metric]
        elif metric in custom_scorers:
            scorer = make_scorer(copy(custom_scorers[metric]))
            scorer.name = scorer._score_func.__name__
        else:
            raise ValueError(
                "Unknown value for the metric parameter, got "
                f"{metric}. Choose from: {', '.join(SCORERS)}."
            )

    elif hasattr(metric, "_score_func"):  # Scoring is a scorer
        scorer = copy(metric)

        # Some scorers use default kwargs
        default_kwargs = ("precision", "recall", "f1", "jaccard")
        if any(name in scorer._score_func.__name__ for name in default_kwargs):
            if not scorer._kwargs:
                scorer._kwargs = {"average": "binary"}

        for key, value in SCORERS.items():
            if scorer.__dict__ == value.__dict__:
                scorer.name = key
                break

    else:  # Scoring is a function with signature metric(y, y_pred)
        scorer = make_scorer(
            score_func=metric,
            greater_is_better=greater_is_better,
            needs_proba=needs_proba,
            needs_threshold=needs_threshold,
        )
        scorer.name = scorer._score_func.__name__

    return scorer


def infer_task(y, goal="class"):
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


def partial_dependence(estimator, X, features):
    """Calculate the partial dependence of features.

    Partial dependence of a feature (or a set of features) corresponds
    to the average response of an estimator for each possible value of
    the feature. Code from sklearn's _partial_dependence.py. Note that
    this implementation always uses method="brute", grid_resolution=100
    and percentiles=(0.05, 0.95).

    Parameters
    ----------
    estimator: class
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


def get_feature_importance(est, attributes=None):
    """Return the feature importance from an estimator.

    Get the feature importance from the provided attribute. For
    meta-estimators, get the mean of the values of the underlying
    estimators.

    Parameters
    ----------
    est: sklearn estimator
        Estimator from which to get the feature importance.

    attributes: sequence or None, default=None
        Attributes to get, in order of importance. If None, use
        score > coefficient > feature importance.

    Returns
    -------
    np.array
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

    return data


# Pipeline functions =============================================== >>

def name_cols(array, original_df, col_names):
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

    col_names: sequence
        Names of the columns used in the transformer.

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


def reorder_cols(df, original_df, col_names):
    """Reorder the columns to their original order.

    This function is necessary in case only a subset of the
    columns in the dataset was used. In that case, we need
    to reorder them to their original order.

    Parameters
    ----------
    df: pd.DataFrame
        Dataset to reorder.

    original_df: pd.DataFrame
        Original dataset (states the order).

    col_names: sequence
        Names of the columns used in the transformer.

    """
    # Check if columns returned by the transformer are already in the dataset
    for col in df:
        if col in original_df and col not in col_names:
            raise ValueError(
                f"Column '{col}' returned by the transformer "
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


def fit_one(transformer, X=None, y=None, message=None, **fit_params):
    """Fit the data using one estimator."""
    X = to_df(X, index=getattr(y, "index", None))
    y = to_series(y, index=getattr(X, "index", None))

    with _print_elapsed_time("Pipeline", message):
        if hasattr(transformer, "fit"):
            args = []
            params = signature(transformer.fit).parameters

            if "X" in params:
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


def transform_one(transformer, X=None, y=None, method="transform"):
    """Transform the data using one estimator."""

    def prepare_df(out):
        """Convert to df and set correct column names and order."""
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
            return reorder_cols(out, X, use_cols)
        else:
            return out

    X = to_df(X, index=getattr(y, "index", None))
    y = to_series(y, index=getattr(X, "index", None))

    args = []
    params = signature(getattr(transformer, method)).parameters
    if "X" in params:
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


def fit_transform_one(transformer, X=None, y=None, message=None, **fit_params):
    """Fit and transform the data using one estimator."""
    fit_one(transformer, X, y, message, **fit_params)
    X, y = transform_one(transformer, X, y)

    return X, y, transformer


def custom_transform(transformer, branch, data=None, verbose=None, method="transform"):
    """Applies a transformer on a branch.

    This function is generic and should work for all
    methods with parameters X and/or y.

    Parameters
    ----------
    transformer: estimator
        Transformer to apply to the data.

    branch: Branch
        Transformer's branch.

    data: tuple or None
        New data to transform on. If tuple, should have form
        (X, y). If None, the transformation is applied directly
        on the branch.

    verbose: int or None, default=None
        Verbosity level for the transformation. If None, the
        estimator's verbosity is used.

    method: str, default="transform"
        Method to apply to the transformer. Choose from: transform
        or inverse_transform.

    """
    # Select provided data or from the branch
    if data:
        X_og, y_og = to_df(data[0]), to_series(data[1])
    else:
        if transformer._train_only:
            X_og, y_og = branch.X_train, branch.y_train
        else:
            X_og, y_og = branch.X, branch.y

    # Adapt the estimator's verbosity
    if verbose is not None:
        if verbose < 0 or verbose > 2:
            raise ValueError(
                "Invalid value for the verbose parameter."
                f"Value should be between 0 and 2, got {verbose}."
            )
        elif hasattr(transformer, "verbose"):
            vb = transformer.verbose  # Save original verbosity
            transformer.verbose = verbose

    if not transformer.__module__.startswith("atom"):
        branch.T.log(f"Applying {transformer.__class__.__name__} to the dataset...", 1)

    X, y = transform_one(transformer, X_og, y_og, method)

    # Apply changes to the branch
    if not data:
        if transformer._train_only:
            branch.train = merge(X, branch.y_train if y is None else y)
        else:
            branch._data = merge(X, branch.y if y is None else y)

            # Since rows can be removed from train and test, reset indices
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

def inverse_transform(self, Xt):
    """Patch function for inverse_transform method.

    Monkey patch for skopt.space.space.Categorical.inverse_transform
    method to fix bug with string and numerical categories combined.

    Parameters
    ----------
    self: Categorical
        Instance from the patched class.

    Xt: array-like, shape=(n_samples,)
        List of categories.

    Returns
    -------
    array-like, shape=(n_samples, n_categories)
        The integer categories.

    """
    return self.transformer.inverse_transform(Xt)


def fit(self, X):
    """Patch function for fit method.

    Monkey patch for skopt.space.transformers.LabelEncoder.fit
    method to fix bug with string and numerical categories combined.

    Parameters
    ----------
    self: LabelEncoder
        Instance from the patched class.

    X: array-like, shape=(n_categories,)
        List of categories.

    """
    self.mapping_ = {v: i for i, v in enumerate(X)}
    self.inverse_mapping_ = {i: v for v, i in self.mapping_.items()}
    return self


def transform(self, X):
    """Patch function for skopt.LabelEncoder transform method.

    Monkey patch for skopt.space.transformers.LabelEncoder.transform
    method to fix bug with string and numerical categories combined.

    Parameters
    ----------
    self: LabelEncoder
        Instance from the patched class.

    X: array-like, shape=(n_samples,)
        List of categories.

    Returns
    -------
    array-like, shape=(n_samples, n_categories)
        The integer categories.

    """
    return [self.mapping_[v] for v in X]


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

def composed(*decs):
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


def crash(f, cache={"last_exception": None}):
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


def method_to_log(f):
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


def plot_from_model(f):
    """If a plot is called from a model, adapt the models parameter."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if hasattr(args[0], "T"):
            return f(args[0].T, args[0].name, *args[1:], **kwargs)
        else:
            return f(*args, **kwargs)

    return wrapper


# Custom scorers =================================================== >>

def true_negatives(y_true, y_pred):
    return int(confusion_matrix(y_true, y_pred).ravel()[0])


def false_positives(y_true, y_pred):
    return int(confusion_matrix(y_true, y_pred).ravel()[1])


def false_negatives(y_true, y_pred):
    return int(confusion_matrix(y_true, y_pred).ravel()[2])


def true_positives(y_true, y_pred):
    return int(confusion_matrix(y_true, y_pred).ravel()[3])


def false_positive_rate(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return float(fp / (fp + tn))


def true_positive_rate(y_true, y_pred):
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tp / (tp + fn))


def true_negative_rate(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return float(tn / (tn + fp))


def false_negative_rate(y_true, y_pred):
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fn / (fn + tp))


# Scorers not implemented by sklearn
CUSTOM_SCORERS = dict(
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


class Predictor(Protocol):
    """Protocol for all predictors."""
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...


class Transformer(Protocol):
    """Protocol for all predictors."""
    def fit(self, X, y, sample_weight=None): ...
    def transform(self, X): ...


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

    def __init__(self, headers, spaces, default_pos="right"):
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
    def to_cell(text, position, space):
        """Get the string format for one cell."""
        if isinstance(text, float):
            text = round(text, 4)
        text = str(text)
        if len(text) > space:
            text = text[:space - 2] + ".."

        if position == "right":
            return text.rjust(space)
        else:
            return text.ljust(space)

    def print_header(self):
        """Print the header line."""
        return self.print({k: k for k in self.headers})

    def print_line(self):
        """Print a line with dashes (usually used after header)."""
        return self.print({k: "-" * s for k, s in zip(self.headers, self.spaces)})

    def print(self, sequence):
        """Convert a sequence to a nice formatted table row."""
        out = []
        for header, pos, space in zip(self.headers, self.positions, self.spaces):
            out.append(self.to_cell(sequence.get(header, "---"), pos, space))

        return "| " + " | ".join(out) + " |"


class PlotCallback:
    """Callback to plot the BO's progress as it runs.

    Parameters
    ----------
    cls: Runner
        Instance from which the callback is called.

    """

    c = 0  # Counter to track which model is being plotted

    def __init__(self, cls):
        self.cls = cls

        # Plot attributes
        max_len = 15  # Maximum steps to show at once in the plot
        self.x, self.y1, self.y2 = {}, {}, {}
        for i in range(len(self.cls._models)):
            self.x[i] = deque(list(range(1, max_len + 1)), maxlen=max_len)
            self.y1[i] = deque([np.NaN for _ in self.x[i]], maxlen=max_len)
            self.y2[i] = deque([np.NaN for _ in self.x[i]], maxlen=max_len)

    def __call__(self, result):
        # Start to fill NaNs with encountered metric values
        if np.isnan(self.y1[self.c]).any():
            for i, value in enumerate(self.y1[self.c]):
                if math.isnan(value):
                    self.y1[self.c][i] = -result.func_vals[-1]
                    if i > 0:  # The first value must remain empty
                        self.y2[self.c][i] = abs(
                            self.y1[self.c][i] - self.y1[self.c][i - 1]
                        )
                    break
        else:  # If no NaNs anymore, continue deque
            self.x[self.c].append(max(self.x[self.c]) + 1)
            self.y1[self.c].append(-result.func_vals[-1])
            self.y2[self.c].append(abs(self.y1[self.c][-1] - self.y1[self.c][-2]))

        if len(result.func_vals) == 1:  # After the 1st iteration, create plot
            self.line1, self.line2, self.ax1, self.ax2 = self.create_figure()
        else:
            self.animate_plot()
            if len(result.func_vals) == self.cls.n_calls:
                self.c += 1  # After the last iteration, go to the next model
                plt.close()

    def create_figure(self):
        """Create the plot.

        Creates a figure with two subplots. The first plot shows the
        score of every trial and the second shows the distance between
        the last consecutive steps.

        """
        plt.ion()  # Call to matplotlib that allows dynamic plotting

        plt.gcf().set_size_inches(10, 8)
        gs = GridSpec(4, 1, hspace=0.05)
        ax1 = plt.subplot(gs[0:3, 0])
        ax2 = plt.subplot(gs[3:4, 0], sharex=ax1)

        # First subplot
        (line1,) = ax1.plot(self.x[self.c], self.y1[self.c], "-o", alpha=0.8)
        ax1.set_title(
            label=f"Bayesian Optimization for {self.cls._models[self.c].fullname}",
            fontsize=self.cls.title_fontsize,
            pad=20,
        )
        ax1.set_ylabel(
            ylabel=self.cls._metric[0].name,
            fontsize=self.cls.label_fontsize,
            labelpad=12,
        )
        ax1.set_xlim(min(self.x[self.c]) - 0.5, max(self.x[self.c]) + 0.5)

        # Second subplot
        (line2,) = ax2.plot(self.x[self.c], self.y2[self.c], "-o", alpha=0.8)
        ax2.set_xlabel(xlabel="Call", fontsize=self.cls.label_fontsize, labelpad=12)
        ax2.set_ylabel(ylabel="d", fontsize=self.cls.label_fontsize, labelpad=12)
        ax2.set_xticks(self.x[self.c])
        ax2.set_xlim(min(self.x[self.c]) - 0.5, max(self.x[self.c]) + 0.5)
        ax2.set_ylim([-0.05, 0.1])

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.xticks(fontsize=self.cls.tick_fontsize)
        plt.yticks(fontsize=self.cls.tick_fontsize)

        return line1, line2, ax1, ax2

    def animate_plot(self):
        """Plot the BO's progress as it runs."""
        self.line1.set_xdata(self.x[self.c])
        self.line1.set_ydata(self.y1[self.c])
        self.line2.set_xdata(self.x[self.c])
        self.line2.set_ydata(self.y2[self.c])
        self.ax1.set_xlim(min(self.x[self.c]) - 0.5, max(self.x[self.c]) + 0.5)
        self.ax2.set_xlim(min(self.x[self.c]) - 0.5, max(self.x[self.c]) + 0.5)
        self.ax1.set_xticks(self.x[self.c])
        self.ax2.set_xticks(self.x[self.c])

        # Adjust y limits if new data goes beyond bounds
        lim = self.line1.axes.get_ylim()
        if np.nanmin(self.y1[self.c]) <= lim[0] or np.nanmax(self.y1[self.c]) >= lim[1]:
            self.ax1.set_ylim(
                [
                    np.nanmin(self.y1[self.c]) - np.nanstd(self.y1[self.c]),
                    np.nanmax(self.y1[self.c]) + np.nanstd(self.y1[self.c]),
                ]
            )
        lim = self.line2.axes.get_ylim()
        if np.nanmax(self.y2[self.c]) >= lim[1]:
            self.ax2.set_ylim(
                [-0.05, np.nanmax(self.y2[self.c]) + np.nanstd(self.y2[self.c])]
            )

        # Pause the data so the figure/axis can catch up
        plt.pause(0.05)


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
    def attr(self):
        """Get the predict_proba, decision_function or predict method if available."""
        for attr in ("predict_proba", "decision_function", "predict"):
            if hasattr(self.T.estimator, attr):
                return attr

    @property
    def explainer(self):
        """Get shap's explainer."""
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

    def get_explanation(self, df, target=1, feature=None, only_one=False):
        """Get an Explanation object.

        Parameters
        ----------
        df: pd.DataFrame
            Data set to look at (subset of the complete dataset).

        target: int, default=1
            Index of the class in the target column to look at.
            Only for multi-class classification tasks.

        feature: int or str
            Index or name of the feature to look at.

        only_one: bool, default=False
            Whether only one row is accepted.

        Returns
        -------
        explanation: shap.Explanation
            Object containing all information (values, base_values, data).

        """
        # Get rows that still need to be calculated
        calculate = df.loc[[i for i in df.index if i not in self._shap_values.index]]
        if not calculate.empty:
            kwargs = {}

            # Minimum of 2 * n_features + 1 evals required (default=500)
            if "max_evals" in signature(self.explainer.__call__).parameters:
                kwargs["max_evals"] = 2 * self.T.n_features + 1

            # Additivity check fails sometimes for no apparent reason
            if "check_additivity" in signature(self.explainer.__call__).parameters:
                kwargs["check_additivity"] = False

            # Calculate the new shap values
            self._explanation = self.explainer(calculate, **kwargs)

            # Remember shap values in the _shap_values attribute
            for i, idx in enumerate(calculate.index):
                self._shap_values.loc[idx] = self._explanation.values[i]

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

        if feature is None:
            return explanation
        else:
            return explanation[:, feature]

    def get_shap_values(self, df, target=1, return_all_classes=False):
        """Get shap values from the Explanation object."""
        values = self.get_explanation(df, target).values
        if return_all_classes:
            if self.T.T.task.startswith("bin") and len(values) != self.T.y.nunique():
                values = [np.array(1 - values), values]

        return values

    def get_interaction_values(self, df):
        """Get shap interaction values from the Explanation object."""
        return self.explainer.shap_interaction_values(df)

    def get_expected_value(self, target=1, return_one=True):
        """Get the expected value of the training set."""
        if self._expected_value is None:
            # Some explainers like Permutation don't have expected_value attr
            if hasattr(self.explainer, "expected_value"):
                self._expected_value = self.explainer.expected_value
            else:
                # The expected value is the average of the model output
                self._expected_value = np.mean(getattr(self.T, f"{self.attr}_train"))

        if return_one and isinstance(self._expected_value, SEQUENCE):
            if len(self._expected_value) == self.T.y.nunique():
                return self._expected_value[target]  # Return target expected value
        elif not return_one and isinstance(self._expected_value, float):
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
        # The sort_dicts parameter is introduced in Python 3.8
        kwargs = {} if sys.version_info[1] < 8 else {"sort_dicts": False}
        return pprint.pformat(dict(self), **kwargs)

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

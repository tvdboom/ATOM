# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing utility constants, functions and classes.

"""

# Standard packages
import math
import logging
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
from collections import deque
from typing import Union, Sequence

# Encoders
from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.basen import BaseNEncoder
from category_encoders.binary import BinaryEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.polynomial import PolynomialEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.woe import WOEEncoder

# Balancers
from imblearn.under_sampling import (
    ClusterCentroids,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    TomekLinks,
)
from imblearn.over_sampling import (
    ADASYN,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
    SMOTE,
    SMOTENC,
    SVMSMOTE,
)

# Sklearn
from sklearn.metrics import SCORERS, get_scorer, make_scorer
from sklearn.utils import _safe_indexing
from sklearn.inspection._partial_dependence import (
    _grid_from_X,
    _partial_dependence_brute,
)

# Plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Global constants ================================================= >>

SEQUENCE = (list, tuple, np.ndarray, pd.Series)

# Variable types
SCALAR = Union[int, float]
SEQUENCE_TYPES = Union[SEQUENCE]
X_TYPES = Union[dict, list, tuple, np.ndarray, pd.DataFrame]
Y_TYPES = Union[int, str, SEQUENCE_TYPES]
TRAIN_TYPES = Union[Sequence[SCALAR], np.ndarray, pd.Series]

# Non-sklearn models
OPTIONAL_PACKAGES = dict(XGB="xgboost", LGB="lightgbm", CatB="catboost")

# List of models that only work for regression/classification tasks
ONLY_CLASS = ["GNB", "MNB", "BNB", "CatNB", "CNB", "LR", "LDA", "QDA"]
ONLY_REG = ["OLS", "Lasso", "EN", "BR", "ARD"]

# List of custom metrics for the scoring method
CUSTOM_METRICS = ["cm", "tn", "fp", "fn", "tp", "lift", "fpr", "tpr", "sup"]

# Acronyms for some common sklearn metrics
METRIC_ACRONYMS = dict(
    ap="average_precision",
    ba="balanced_accuracy",
    auc="roc_auc",
    ev="explained_variance",
    me="max_error",
    mae="neg_mean_absolute_error",
    mse="neg_mean_squared_error",
    rmse="neg_root_mean_squared_error",
    msle="neg_mean_squared_log_error",
    medae="neg_median_absolute_error",
    poisson="neg_mean_poisson_deviance",
    gamma="neg_mean_gamma_deviance",
)

# All available encoding strategies
ENCODER_TYPES = dict(
    backwarddifference=BackwardDifferenceEncoder,
    basen=BaseNEncoder,
    binary=BinaryEncoder,
    catboost=CatBoostEncoder,
    # hashing=HashingEncoder,
    helmert=HelmertEncoder,
    jamesstein=JamesSteinEncoder,
    leaveoneout=LeaveOneOutEncoder,
    mestimate=MEstimateEncoder,
    # onehot=OneHotEncoder,
    ordinal=OrdinalEncoder,
    polynomial=PolynomialEncoder,
    sum=SumEncoder,
    target=TargetEncoder,
    woe=WOEEncoder,
)

# All available balancing strategies
BALANCER_TYPES = dict(
    clustercentroids=ClusterCentroids,
    condensednearestneighbour=CondensedNearestNeighbour,
    editednearestneighborus=EditedNearestNeighbours,
    repeatededitednearestneighbours=RepeatedEditedNearestNeighbours,
    allknn=AllKNN,
    instancehardnessthreshold=InstanceHardnessThreshold,
    nearmiss=NearMiss,
    neighbourhoodcleaningrule=NeighbourhoodCleaningRule,
    onesidedselection=OneSidedSelection,
    randomundersampler=RandomUnderSampler,
    tomeklinks=TomekLinks,
    adasyn=ADASYN,
    borderlinesmote=BorderlineSMOTE,
    kmanssmote=KMeansSMOTE,
    randomoversampler=RandomOverSampler,
    smote=SMOTE,
    smotenc=SMOTENC,
    svmsmote=SVMSMOTE,
)


# Functions ======================================================== >>

def flt(item):
    """Return value if item is sequence of length 1."""
    return item[0] if isinstance(item, SEQUENCE) and len(item) == 1 else item


def lst(item):
    """Return list if item is not a sequence."""
    return [item] if not isinstance(item, SEQUENCE) else item


def it(item):
    """Return int if it's a rounded float, else item."""
    try:
        is_equal = int(item) == float(item)
    except ValueError:
        return item

    return int(item) if is_equal else item


def divide(a, b):
    """Divide two numbers and return 0 if division by zero."""
    return np.divide(a, b) if b != 0 else 0


def merge(X, y):
    """Merge a pd.DataFrame and pd.Series into one dataframe."""
    return X.merge(y.to_frame(), left_index=True, right_index=True)


def catch_return(args):
    """Returns always two arguments independent of length arrays."""
    if not isinstance(args, tuple):
        return args, None
    else:
        return args[0], args[1]


def variable_return(X, y):
    """Return one or two arguments depending on if y is None."""
    if y is None:
        return X
    else:
        return X, y


def check_dim(cls, method):
    """Raise an error if it's a deep learning dataset."""
    if list(cls.X.columns) == ["Features"]:
        raise PermissionError(
            f"The {method} method is not available for deep learning datasets!"
        )


def check_goal(cls, method, goal):
    """Raise an error if the goal is invalid."""
    if not cls.goal == goal:
        raise PermissionError(
            f"The {method} method is only available for {goal} tasks!"
        )


def check_binary_task(cls, method):
    """Raise an error if the task is invalid."""
    if not cls.task.startswith("bin"):
        raise PermissionError(
            f"The {method} method is only available for binary classification tasks!"
        )


def check_predict_proba(models, method):
    """Raise an error if a model doesn't have a predict_proba method."""
    for m in models:
        if not hasattr(m.estimator, "predict_proba"):
            raise AttributeError(
                f"The {method} method is only available "
                f"for models with a predict_proba method, got {m}."
            )


def check_scaling(X):
    """Check if the provided data is scaled to mean=0 and std=1."""
    mean = X.mean(axis=1).mean()
    std = X.std(axis=1).mean()
    return True if mean < 0.05 and 0.5 < std < 1.5 else False


def get_best_score(item, metric=0):
    """Returns the bagging or test score of a model.

    Parameters
    ----------
    item: model or pd.Series
        Model instance or row from the results dataframe.

    metric: int, optional (default=0)
        Index of the metric to use.

    """
    if item.mean_bagging:
        return lst(item.mean_bagging)[metric]
    else:
        return lst(item.metric_test)[metric]


def time_to_string(t_init):
    """Convert time integer to string.

    Convert a time duration to a string of format 00h:00m:00s
    or 1.000s if under 1 min.

    Parameters
    ----------
    t_init: float
        Time to convert (in seconds).

    Returns
    -------
    time: str
        Time representation.

    """
    t = time() - t_init  # Total time in seconds
    h = int(t / 3600.0)
    m = int(t / 60.0) - h * 60
    s = t - h * 3600 - m * 60
    if h < 1 and m < 1:  # Only seconds
        return f"{s:.3f}s"
    elif h < 1:  # Also minutes
        return f"{m}m:{int(s):02}s"
    else:  # Also hours
        return f"{h}h:{m:02}m:{int(s):02}s"


def to_df(data, index=None, columns=None, pca=False):
    """Convert a dataset to pd.Dataframe.

    Parameters
    ----------
    data: sequence
        Dataset to convert to a dataframe.

    index: sequence or Index
        Values for the dataframe's index.

    columns: sequence or None, optional(default=None)
        Name of the columns. If None, the names are autofilled.

    pca: bool, optional (default=False)
        Whether the columns are Features or Components.

    Returns
    -------
    df: pd.DataFrame
        Transformed dataframe.

    """
    if not isinstance(data, pd.DataFrame):
        if columns is None and not pca:
            columns = [f"Feature {str(i)}" for i in range(len(data[0]))]
        elif columns is None:
            columns = [f"Component {str(i)}" for i in range(len(data[0]))]
        data = pd.DataFrame(data, index=index, columns=columns)

    return data


def to_series(data, index=None, name="target"):
    """Convert a column to pd.Series.

    Parameters
    ----------
    data: list, tuple or np.ndarray
        Data to convert.

    index: array or Index, optional (default=None)
        Values for the series" index.

    name: string, optional (default="target")
        Name of the target column.

    Returns
    -------
    series: pd.Series
        Transformed series.

    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data, index=index, name=name)

    return data


def arr(df):
    """From dataframe to multidimensional array.

    When the data consist of more than 2 dimensions, ATOM stores
    it in a df with a single column, "Features". This function
    extracts the arrays from every row and returns them stacked.

     """
    if list(df.columns) == ["Features"]:
        return np.stack(df["Features"].values)
    else:
        return df


def prepare_logger(logger, class_name):
    """Prepare logging file.

    Parameters
    ----------
    logger: str, class or None
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. Use "auto" for default name.
        - If class: Python `Logger` object.

        The default name created consists of the class' name
        followed by the timestamp of the logger's creation.

    class_name: str
        Name of the class from which the function is called.
        Used for default name creation when log="auto".

    Returns
    -------
    logger: class
        Logger object.

    """
    if not logger:  # Empty string or None
        return

    elif isinstance(logger, str):
        # Prepare the FileHandler's name
        if not logger.endswith(".log"):
            logger += ".log"
        if logger == "auto.log" or logger.endswith("/auto.log"):
            current = datetime.now().strftime("%d%b%y_%Hh%Mm%Ss")
            logger = logger.replace("auto", class_name + "_" + current)

        # Define file handler and set formatter
        file_handler = logging.FileHandler(logger)
        formatter = logging.Formatter("%(asctime)s: %(message)s")
        file_handler.setFormatter(formatter)

        # Define logger
        logger = logging.getLogger(class_name + "_logger")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if logger.hasHandlers():  # Remove existing handlers
            logger.handlers.clear()
        logger.addHandler(file_handler)  # Add file handler to logger

    elif type(logger) != logging.Logger:  # Should be python "Logger" object"
        raise TypeError(
            "Invalid value for the logger parameter. Should be a "
            f"python logging.Logger object, got {type(logger)}!"
        )

    return logger


def check_is_fitted(estimator, attributes=None, msg=None):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (not None or empty). Otherwise it raises a
    NotFittedError.

    Parameters
    ----------
    estimator: class
        Class instance for which the check is performed.

    attributes: str, sequence or None, optional (default=None)
        Attribute(s) to check. If None, the estimator is considered
        fitted if there exist an attribute that ends with a underscore
        and does not start with double underscore.

    msg: str, optional (default=None)
        Default error message.

    """

    def check_attr(attr):
        """Return empty pandas or None/empty sequence."""
        if isinstance(getattr(estimator, attr), (pd.DataFrame, pd.Series)):
            return getattr(estimator, attr).empty
        else:
            return not getattr(estimator, attr)

    if msg is None:
        msg = (
            f"This {type(estimator).__name__} instance is not fitted yet. Call "
            "'fit' or 'run' with appropriate arguments before using this estimator."
        )

    if all([check_attr(attr) for attr in lst(attributes)]):
        raise NotFittedError(msg)


def fit_init(est_params, fit=False):
    """Return the est_params for the __init__ or fit method.

    Select parameters that end with _fit for the fit method,
    else they're used when initializing the instance.

    Parameters
    ----------
    est_params: dict
        Parameters that need to be classified.

    fit: bool, optional (default=False)
        Whether the parameters are for the fit method.

    """
    if fit:
        return {k[:-4]: v for k, v in est_params.items() if k.endswith("_fit")}
    else:
        return {k: v for k, v in est_params.items() if not k.endswith("_fit")}


def get_acronym(model):
    """Get the right model acronym.

    Parameters
    ----------
    model: str
        Acronym of the model, case insensitive.

    Returns
    -------
    name: str
        Correct model name acronym as present in MODEL_LIST.

    """
    # Not imported on top of file because of module interconnection
    from .models import MODEL_LIST

    for acronym in MODEL_LIST:
        if model.lower() == acronym.lower():
            return acronym

    raise ValueError(
        f"Unknown model: {model}! Choose from: {', '.join(MODEL_LIST)}."
    )


def get_metric(metric, greater_is_better, needs_proba, needs_threshold):
    """Get the right metric depending on the input type.

    Parameters
    ----------
    metric: str or callable
        Metric as a string, function or scorer.

    greater_is_better: bool
        whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower is
        better. Will be ignored if the metric is a string or a scorer.

    needs_proba: bool
        Whether the metric function requires probability estimates of
        a classifier. Is ignored if the metric is a string or a scorer.

    needs_threshold: bool
        Whether the metric function takes a continuous decision
        certainty. Is ignored if the metric is a string or a scorer.

    Returns
    -------
    scorer: callable
        Scorer object.

    """

    def get_scorer_name(scorer):
        """Return the name of the provided scorer."""
        for key, value in SCORERS.items():
            if scorer.__dict__ == value.__dict__:
                return key

    if isinstance(metric, str):
        if metric.lower() in METRIC_ACRONYMS:
            metric = METRIC_ACRONYMS[metric.lower()]
        elif metric not in SCORERS:
            raise ValueError(
                "Unknown value for the metric parameter, got "
                f"{metric}. Try one of: {', '.join(SCORERS)}."
            )
        metric = get_scorer(metric)
        metric.name = get_scorer_name(metric)

    elif hasattr(metric, "_score_func"):  # Provided metric is scoring
        metric.name = get_scorer_name(metric)

    else:  # Metric is a function with signature metric(y, y_pred)
        metric = make_scorer(
            score_func=metric,
            greater_is_better=greater_is_better,
            needs_proba=needs_proba,
            needs_threshold=needs_threshold,
        )
        metric.name = metric._score_func.__name__

    return metric


def get_default_metric(task):
    """Return the default metric for each task.

    Parameters
    ----------
    task: str
        One of binary classification, multiclass classification or
        regression.

    """
    if task.startswith("bin"):
        return get_metric("f1", True, False, False)
    elif task.startswith("multi"):
        return get_metric("f1_weighted", True, False, False)
    else:
        return get_metric("r2", True, False, False)


def infer_task(y, goal="classification"):
    """Infer the task corresponding to a target column.

    If goal is provided, only look at number of unique values to
    determine the classification task.

    Parameters
    ----------
    y: pd.Series
        Target column from which to infer the task.

    goal: str, optional (default="classification")
        Classification or regression goal.

    Returns
    -------
    task: str
        Inferred task.

    """
    if goal == "regression":
        return goal

    unique = y.unique()
    if len(unique) == 1:
        raise ValueError(f"Only found 1 target value: {unique[0]}")
    elif len(unique) == 2:
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
    estimator : class
        Model estimator to use.

    X : pd.DataFrame
        Feature set used to generate a grid of values for the target
        features (where the partial dependence will be evaluated), and
        also to generate values for the complement features.

    features : int, str or sequence
        The feature or pair of interacting features for which the
        partial dependency should be computed.

    Returns
    -------
    averaged_predictions: np.ndarray
        Average of the predictions.

    values: list
        Values used for the predictions.

    """
    grid, values = _grid_from_X(_safe_indexing(X, features, axis=1), (0.05, 0.95), 100)

    averaged_predictions = _partial_dependence_brute(
        estimator, grid, features, X, "auto"
    )

    # Reshape averaged_predictions to (n_outputs, n_values_feature_0, ...)
    averaged_predictions = averaged_predictions.reshape(
        -1, *[val.shape[0] for val in values]
    )

    return averaged_predictions, values


# Functions shared by classes ======================================= >>

def transform(est_branch, X, y, verbose, **kwargs):
    """Transform new data through all transformers in a branch.

    The outliers and balance transformations are not included by
    default since they should only be applied on the training set.

    Parameters
    ----------
    est_branch: pd.Series
        Estimators in a branch.

    X: dict, sequence, np.ndarray or pd.DataFrame
        Feature set with shape=(n_samples, n_features).

    y: int, str, sequence, np.ndarray, pd.Series or None
        - If None: y is not used in the transformation.
        - If int: Index of the target column in X.
        - If str: Name of the target column in X.
        - Else: Target column with shape=(n_samples,).

    verbose: int
        Verbosity level of the transformers.

    **kwargs
        Additional keyword arguments to customize which transforming
        methods to apply. You can either select them via their index,
        e.g. pipeline = [0, 1, 4] or include/exclude them via every
        individual transformer, e.g. impute=True, encode=False.

    Returns
    -------
    X: pd.DataFrame
        Transformed dataset.

    y: pd.Series
        Transformed target column. Only returned if provided.

    """

    def transform_one(est):
        """Transform a single estimator."""
        nonlocal X, y

        vb = est.get_params()["verbose"]  # Save original verbosity
        est.verbose = verbose

        # Some transformers return no y, but we need the original
        X, y_returned = catch_return(est.transform(X, y))
        y = y if y_returned is None else y_returned

        # Reset the original verbosity
        est.verbose = vb

    # Check verbose parameter
    if verbose < 0 or verbose > 2:
        raise ValueError(
            "Invalid value for the verbose parameter."
            f"Value should be between 0 and 2, got {verbose}."
        )

    # Data cleaning and feature engineering methods and their classes
    steps = dict(
        cleaner="Cleaner",
        scale="Scaler",
        impute="Imputer",
        encode="Encoder",
        outliers="Outliers",
        balance="Balancer",
        feature_generation="FeatureGenerator",
        feature_selection="FeatureSelector",
    )

    # Set default values if pipeline is not provided
    if kwargs.get("pipeline") is None:
        for key, value in steps.items():
            if key not in kwargs:
                kwargs[value] = False if key in ["outliers", "balance"] else True
            else:
                kwargs[value] = kwargs.pop(key)

    # Transform either one or all the transformers in the pipeline
    for i, est in enumerate(est_branch):
        if i in kwargs.get("pipeline", []) or kwargs.get(est.__class__.__name__):
            if kwargs.get("_one_trans", i) == i:
                transform_one(est)

    return variable_return(X, y)


def delete(self, models):
    """Delete models from a trainer's pipeline.

    Removes all traces of a model in the pipeline (except for the
    `errors` attribute). If the winning model is removed. The next
    best model (through metric_test or mean_bagging if available)
    is selected as winner.If all models are removed, the metric and
    approach are reset. Use this method to drop unwanted models from
    the pipeline or to clear memory before saving the instance.

    Parameters
    ----------
    self: class
        Trainer for which to delete the model.

    models: sequence
        Name of the models to delete from the pipeline.

    """
    for model in models:
        self.models.remove(model)

        # Remove the model from the results dataframe
        if isinstance(self._results.index, pd.MultiIndex):
            self._results = self._results.iloc[
                ~self._results.index.get_level_values(1).str.contains(model)
            ]
        else:
            self._results.drop(model, axis=0, inplace=True, errors="ignore")

        # If no models, reset the metric and the training approach
        if not self.models:
            self.metric_ = []
            if hasattr(self, "_approach"):
                self._approach = None

        del self.__dict__[model]
        del self.__dict__[model.lower()]


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
    catch or multiple calls to crash), its not re-written in the logger.

    """

    def wrapper(*args, **kwargs):
        logger = args[0].logger if hasattr(args[0], "logger") else args[0].T.logger

        if logger is not None:
            try:  # Run the function
                return f(*args, **kwargs)

            except Exception as exception:
                # If exception is not same as last, write to log
                if exception is not cache["last_exception"]:
                    cache["last_exception"] = exception
                    logger.exception("Exception encountered:")

                raise exception  # Always raise it
        else:
            return f(*args, **kwargs)

    return wrapper


def method_to_log(f):
    """Save called functions to log file."""

    def wrapper(*args, **kwargs):
        # Get logger (for model subclasses called from BasePredictor)
        logger = args[0].logger if hasattr(args[0], "logger") else args[0].T.logger

        if logger is not None:
            if f.__name__ != "__init__":
                logger.info("")
            logger.info(f"{args[0].__class__.__name__}.{f.__name__}()")

        result = f(*args, **kwargs)
        return result

    return wrapper


def plot_from_model(f):
    """If a plot is called from a model, adapt the models parameter."""

    def wrapper(*args, **kwargs):
        if hasattr(args[0], "T"):
            return f(args[0].T, args[0].name, *args[1:], **kwargs)
        else:
            return f(*args, **kwargs)

    return wrapper


# Classes ========================================================== >>

class NotFittedError(ValueError, AttributeError):
    """Exception called when the instance is not yet fitted."""
    pass


class PlotCallback(object):
    """Callback to plot the BO's progress as it runs.

    Parameters
    ----------
    M: class
        Model subclass.

    """

    def __init__(self, *args):
        self.M = args[0]

        # Plot attributes
        max_len = 15  # Maximum steps to show at once in the plot
        self.x = deque(list(range(1, max_len + 1)), maxlen=max_len)
        self.y1 = deque([np.NaN for _ in self.x], maxlen=max_len)
        self.y2 = deque([np.NaN for _ in self.x], maxlen=max_len)

    def __call__(self, result):
        # Start to fill NaNs with encountered metric values
        if np.isnan(self.y1).any():
            for i, value in enumerate(self.y1):
                if math.isnan(value):
                    self.y1[i] = -result.func_vals[-1]
                    if i > 0:  # The first value must remain empty
                        self.y2[i] = abs(self.y1[i] - self.y1[i - 1])
                    break
        else:  # If no NaNs anymore, continue deque
            self.x.append(max(self.x) + 1)
            self.y1.append(-result.func_vals[-1])
            self.y2.append(abs(self.y1[-1] - self.y1[-2]))

        if len(result.func_vals) == 1:  # After the 1st iteration, create plot
            self.line1, self.line2, self.ax1, self.ax2 = self.create_figure()
        else:
            self.animate_plot()

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
        (line1,) = ax1.plot(self.x, self.y1, "-o", alpha=0.8)
        ax1.set_title(
            label=f"Bayesian Optimization for {self.M.fullname}",
            fontsize=self.M.T.title_fontsize,
            pad=20,
        )
        ax1.set_ylabel(
            ylabel=self.M.T.metric_[0].name,
            fontsize=self.M.T.label_fontsize,
            labelpad=12,
        )
        ax1.set_xlim(min(self.x) - 0.5, max(self.x) + 0.5)

        # Second subplot
        (line2,) = ax2.plot(self.x, self.y2, "-o", alpha=0.8)
        ax2.set_xlabel(xlabel="Call", fontsize=self.M.T.label_fontsize, labelpad=12)
        ax2.set_ylabel(ylabel="d", fontsize=self.M.T.label_fontsize, labelpad=12)
        ax2.set_xticks(self.x)
        ax2.set_xlim(min(self.x) - 0.5, max(self.x) + 0.5)
        ax2.set_ylim([-0.05, 0.1])

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.xticks(fontsize=self.M.T.tick_fontsize)
        plt.yticks(fontsize=self.M.T.tick_fontsize)

        return line1, line2, ax1, ax2

    def animate_plot(self):
        """Plot the BO's progress as it runs."""
        self.line1.set_xdata(self.x)
        self.line1.set_ydata(self.y1)
        self.line2.set_xdata(self.x)
        self.line2.set_ydata(self.y2)
        self.ax1.set_xlim(min(self.x) - 0.5, max(self.x) + 0.5)
        self.ax2.set_xlim(min(self.x) - 0.5, max(self.x) + 0.5)
        self.ax1.set_xticks(self.x)
        self.ax2.set_xticks(self.x)

        # Adjust y limits if new data goes beyond bounds
        lim = self.line1.axes.get_ylim()
        if np.nanmin(self.y1) <= lim[0] or np.nanmax(self.y1) >= lim[1]:
            self.ax1.set_ylim([
                np.nanmin(self.y1) - np.nanstd(self.y1),
                np.nanmax(self.y1) + np.nanstd(self.y1),
            ])
        lim = self.line2.axes.get_ylim()
        if np.nanmax(self.y2) >= lim[1]:
            self.ax2.set_ylim([-0.05, np.nanmax(self.y2) + np.nanstd(self.y2)])

        # Pause the data so the figure/axis can catch up
        plt.pause(0.01)

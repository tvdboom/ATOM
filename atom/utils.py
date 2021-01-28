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
from functools import wraps
from datetime import datetime
from inspect import signature
from collections import deque
from typing import Union, Sequence
from collections.abc import MutableMapping

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
    logloss="neg_log_loss",
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
    for m in [m for m in models if m.name != "Vote"]:
        if not hasattr(m.estimator, "predict_proba"):
            raise AttributeError(
                f"The {method} method is only available for "
                f"models with a predict_proba method, got {m.name}."
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
    if getattr(item, "mean_bagging", None):
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
    h = t // 3600
    m = t / 60 - h * 60
    s = t - h * 3600 - m * 60
    if h < 1 and m < 1:  # Only seconds
        return f"{s:.3f}s"
    elif h < 1:  # Also minutes
        return f"{m}m:{s:02}s"
    else:  # Also hours
        return f"{h}h:{m:02}m:{s:02}s"


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

        The default name consists of the class' name followed by
        the timestamp of the logger's creation.

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
            "Invalid value for the logger parameter. Expected a "
            f"logging.Logger object, got {type(logger)}!"
        )

    return logger


def check_is_fitted(estimator, exception=True, attributes=None):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (not None or empty). Otherwise it raises a
    NotFittedError. Extension on sklearn's function that accounts
    for empty dataframes and series.

    Parameters
    ----------
    estimator: class
        Class instance for which the check is performed.

    exception: bool, optional (default=True)
        Whether to raise an exception if the estimator is not fitted.
        If False, it returns False instead.

    attributes: str, sequence or None, optional (default=None)
        Attribute(s) to check. If None, the estimator is considered
        fitted if there exist an attribute that ends with a underscore
        and does not start with double underscore.

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
    elif not all([check_attr(attr) for attr in lst(attributes)]):
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


def get_acronym(model, must_be_equal=True):
    """Get the right model acronym.

    Parameters
    ----------
    model: str
        Acronym of the model, case-insensitive.

    must_be_equal: bool, optional (default=True)
        Whether the model name must be exactly equal or start
        with the acronym.

    Returns
    -------
    name: str
        Correct model name acronym as present in MODEL_LIST.

    """
    # Not imported on top of file because of module interconnection
    from .models import MODEL_LIST

    for acronym in MODEL_LIST:
        cond_1 = must_be_equal and model.lower() == acronym.lower()
        cond_2 = not must_be_equal and model.lower().startswith(acronym.lower())
        if cond_1 or cond_2:
            return acronym

    raise ValueError(
        f"Unknown model: {model}! Choose from: {', '.join(MODEL_LIST)}."
    )


def get_metric(metric, gib=True, needs_proba=False, needs_threshold=False):
    """Get the right metric depending on the input type.

    Parameters
    ----------
    metric: str or callable
        Metric as a string, function or scorer.

    gib: bool, optional (default=True)
        whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower is
        better. Will be ignored if the metric is a string or a scorer.

    needs_proba: bool, optional (default=False)
        Whether the metric function requires probability estimates of
        a classifier. Is ignored if the metric is a string or a scorer.

    needs_threshold: bool, optional (default=False)
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
            greater_is_better=gib,
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
        return {"f1": get_metric("f1", True, False, False)}
    elif task.startswith("multi"):
        return {"f1_weighted": get_metric("f1_weighted", True, False, False)}
    else:
        return {"r2": get_metric("r2", True, False, False)}


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


def names_from_estimator(cls, estimator):
    """Get the model's acronym and fullname from an estimator.

    Parameters
    ----------
    cls: class
        Trainer from which the function is called.

    estimator: class
        Estimator instance to get the information from.

    Returns
    -------
    acronym: str
        Model's acronym.

    fullname: str
        Model's complete name.

    """
    from .models import MODEL_LIST

    get_name = lambda est: est.__class__.__name__
    for key, value in MODEL_LIST.items():
        model = value(cls)
        if get_name(model.get_estimator()) == get_name(estimator):
            return model.acronym, model.fullname


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

    averaged_predictions, _ = _partial_dependence_brute(
        estimator, grid, features, X, "auto"
    )

    # Reshape averaged_predictions to (n_outputs, n_values_feature_0, ...)
    averaged_predictions = averaged_predictions.reshape(
        -1, *[val.shape[0] for val in values]
    )

    return averaged_predictions, values


# Functions shared by classes ======================================= >>

def add_transformer(self, transformer, name=None, train_only=False):
    """Add a transformer to the current branch.

    If the transformer is not fitted, it is fitted on the
    complete training set. Afterwards, the data set is
    transformed and the transformer is added to atom's
    pipeline.

    If the transformer has the n_jobs and/or random_state
    parameters and they are left to their default value,
    they adopt atom's values.

    Parameters
    ----------
    self: class
        Instance of atom from which the function is called.

    transformer: class
        Transformer to add to the pipeline. Should implement a
        `transform` method.

    name: str or None, optional (default=None)
        Name of the transformation step. If None, it defaults to
        the __name__ of the transformer's class (lower case).

    train_only: bool, optional (default=False)
        Whether to apply the transformer only on the train set or
        on the complete dataset.

    """
    def get_cols(array, dataframe):
        """Get the column names after a transformation.

        If the number of columns is unchanged, the original
        column names are returned. Else, give the column a
        default name if the column values changed.

        Parameters
        ----------
        array: np.ndarray
            Transformed dataset.

        dataframe: pd.DataFrame
            Original dataset.

        """
        # If columns were only transformed, return og names
        if array.shape[1] == dataframe.shape[1]:
            return dataframe.columns

        # If columns were added or removed
        cols = []
        for i, col in enumerate(array.T):
            mask = dataframe.apply(lambda c: all(c == col))
            if any(mask):
                cols.append(mask[mask].index.values[0])
            else:
                cols.append(f"Feature {str(i)}")

        return cols

    if not hasattr(transformer, "transform"):
        raise ValueError("The transformer should have a transform method!")

    # Add BaseTransformer params to the estimator if left to default
    if all(hasattr(transformer, attr) for attr in ["get_params", "set_params"]):
        sign = signature(transformer.__init__).parameters
        for p in ["n_jobs", "random_state"]:
            if p in sign and transformer.get_params()[p] == sign[p]._default:
                transformer.set_params(**{p: getattr(self, p)})

    if hasattr(transformer, "fit") and not check_is_fitted(transformer, False):
        transformer.fit(self.X_train, self.y_train)

    if train_only:
        if "y" in signature(transformer.transform).parameters:
            X, y = catch_return(transformer.transform(self.X_train, self.y_train))
        else:
            X, y = catch_return(transformer.transform(self.X_train))

        if not isinstance(X, pd.DataFrame):
            X = to_df(X, columns=get_cols(X, self.X_train))
        if y is None:
            self.train = merge(X, self.y_train)
        else:
            self.train = merge(X, to_series(y, name=self.target))

    else:
        if "y" in signature(transformer.transform).parameters:
            X, y = catch_return(transformer.transform(self.X, self.y))
        else:
            X, y = catch_return(transformer.transform(self.X))

        if not isinstance(X, pd.DataFrame):
            X = to_df(X, columns=get_cols(X, self.X))
        if y is None:
            self.dataset = merge(X, self.y)
        else:
            self.dataset = merge(X, to_series(y, name=self.target))

        # Since rows can be removed from train and test, reset indices
        self.branch.idx[1] = len(X[X.index >= self.branch.idx[0]])
        self.branch.idx[0] = len(X[X.index < self.branch.idx[0]])

    self.dataset = self.dataset.reset_index(drop=True)

    # Add the train_only parameter as attr to the transformer
    transformer.train_only = train_only

    # Add the transformer to the pipeline
    self.branch.pipeline = self.pipeline.append(
        pd.Series(
            data=[transformer],
            index=[name if name else transformer.__class__.__name__.lower()],
            name=self._current,
        )
    )


def transform(est_branch, X, y, verbose, **kwargs):
    """Transform new data through all transformers in a branch.

    The outliers and balance transformations are not included by
    default since they should only be applied on the training set.

    Parameters
    ----------
    est_branch: pd.Series
        Estimators in a branch.

    X: dict, list, tuple, np.ndarray or pd.DataFrame
        Feature set with shape=(n_samples, n_features).

    y: int, str, sequence or None
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

    # Transform either one or all the transformers in the pipeline
    p = kwargs.get("pipeline", [])
    for i, (idx, est) in enumerate(est_branch.iteritems()):
        is_default = not p and not est.train_only and kwargs.get(idx) is None
        if i in p or idx in p or kwargs.get(idx) or is_default:
            if kwargs.get("_one_trans", i) == i:
                transform_one(est)

    return variable_return(X, y)


def delete(self, models):
    """Delete models from a trainer's pipeline.

    Removes all traces of a model in the pipeline (except for the
    `errors` attribute). If all models are removed, the metric and
    approach are reset.

    Parameters
    ----------
    self: class
        Trainer for which to delete the model.

    models: sequence
        Name of the models to delete from the pipeline.

    """
    for model in models:
        self._models.pop(model.lower())

    # If no models, reset the metric
    if not self._models:
        self._metric = {}


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
    @wraps(f)
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
    @wraps(f)
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
    @wraps(f)
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


class PlotCallback:
    """Callback to plot the BO's progress as it runs.

    Parameters
    ----------
    cls: class
        Trainer from which the callback is called.

    """

    def __init__(self, cls):
        self.cls = cls

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
            label=f"Bayesian Optimization performance",
            fontsize=self.cls.title_fontsize,
            pad=20,
        )
        ax1.set_ylabel(
            ylabel=self.cls._metric[0].name,
            fontsize=self.cls.label_fontsize,
            labelpad=12,
        )
        ax1.set_xlim(min(self.x) - 0.5, max(self.x) + 0.5)

        # Second subplot
        (line2,) = ax2.plot(self.x, self.y2, "-o", alpha=0.8)
        ax2.set_xlabel(xlabel="Call", fontsize=self.cls.label_fontsize, labelpad=12)
        ax2.set_ylabel(ylabel="d", fontsize=self.cls.label_fontsize, labelpad=12)
        ax2.set_xticks(self.x)
        ax2.set_xlim(min(self.x) - 0.5, max(self.x) + 0.5)
        ax2.set_ylim([-0.05, 0.1])

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.xticks(fontsize=self.cls.tick_fontsize)
        plt.yticks(fontsize=self.cls.tick_fontsize)

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


class CustomDict(MutableMapping):
    """Custom ordered dictionary.

    Custom dictionary for the _models and _metric private attributes
    of the trainers. the main differences with the Python dictionary
    are:
        - It has ordered entries.
        - It allows getting an item from an index position.
        - It can insert key value pairs at a specific position.
        - Keys are lower case and requests are case insensitive.
        - If iterated over, it iterates over its values, not the keys!

    """

    @staticmethod
    def _conv(key):
        return key.lower() if isinstance(key, str) else key

    def __init__(self, iterable_or_mapping=None, **kwargs):
        """Class initializer.

        Mimics a dictionary's initialization and accepts the same
        arguments. You have to pass an ordered iterable or mapping
        unless you want the order to be arbitrary.

        """
        self.__data = {}
        self.__keys = []

        if iterable_or_mapping is not None:
            try:
                iterable = iterable_or_mapping.items()
            except AttributeError:
                iterable = iterable_or_mapping

            for key, value in iterable:
                self.__keys.append(self._conv(key))
                self.__data[self._conv(key)] = value

        for key, value in kwargs.items():
            self.__keys.append(self._conv(key))
            self.__data[self._conv(key)] = value

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.__data[key.lower()]
        else:
            return self.__data[self.__keys[key]]

    def __setitem__(self, key, value):
        self.__keys.append(self._conv(key))
        self.__data[self._conv(key)] = value

    def __delitem__(self, key):
        del self.__data[self._conv(key)]
        self.__keys.remove(self._conv(key))

    def __iter__(self):
        yield from self.values()

    def __len__(self):
        return len(self.__keys)

    def __contains__(self, key):
        return key in self.__keys

    def __repr__(self):
        return str(dict(self))

    def keys(self):
        yield from self.__keys

    def items(self):
        for key in self.__keys:
            yield key, self.__data[key]

    def values(self):
        for key in self.__keys:
            yield self.__data[key]

    def insert(self, pos, new_key, value):
        if isinstance(pos, str):
            pos = self.__keys.index(self._conv(pos))
        try:
            self.__keys.insert(pos, self._conv(new_key))
            self.__data[self._conv(new_key)] = value
        except ValueError:
            raise KeyError(pos) from ValueError

    def get(self, key, default=None):
        try:
            return self.__data[self._conv(key)]
        except KeyError:
            return default

    def pop(self, key, default=None):
        value = self.get(key)
        if value:
            self.__delitem__(key)
            return value
        else:
            return default

    def popitem(self):
        try:
            return self.__data.pop(self.__keys.pop())
        except IndexError:
            raise KeyError(f"{self.__class__.__name__} is empty")

    def clear(self):
        self.__keys = []
        self.__data = {}

    def update(self, mapping={}, **kwargs):
        for key, value in mapping.items():
            if self._conv(key) not in self.__keys:
                self.__keys.append(self._conv(key))
            self.__data[self._conv(key)] = value

    def setdefault(self, key, default=None):
        try:
            return self[self._conv(key)]
        except KeyError:
            self[self._conv(key)] = default
            return self[self._conv(key)]

    def index(self, key):
        return self.__keys.index(self._conv(key))

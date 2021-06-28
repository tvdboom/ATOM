# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: Mavs
Description: Module containing utility constants, functions and classes.

"""

# Standard packages
import math
import logging
import numpy as np
import pandas as pd
from typing import Union
from functools import wraps
from collections import deque
from datetime import datetime
from inspect import signature
from collections.abc import MutableMapping
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
)
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, OPTICS

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
from sklearn.metrics import (
    SCORERS,
    make_scorer,
    confusion_matrix,
    matthews_corrcoef,
)
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

# Non-sklearn models
OPTIONAL_PACKAGES = dict(XGB="xgboost", LGB="lightgbm", CatB="catboost")

# List of models that only work for regression/classification tasks
ONLY_CLASS = ["GNB", "MNB", "BNB", "CatNB", "CNB", "LR", "LDA", "QDA"]
ONLY_REG = ["OLS", "Lasso", "EN", "BR", "ARD"]

# Attributes shared betwen atom and a pd.DataFrame
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
)

# List of available distributions
DISTRIBUTIONS = (
    "beta",
    "expon",
    "gamma",
    "invgauss",
    "lognorm",
    "norm",
    "pearson3",
    "triang",
    "uniform",
    "weibull_min",
    "weibull_max",
)

# List of custom metrics for the scoring method
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

# Acronyms for some of the common scorers
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

# All available scaling strategies
SCALING_STRATS = dict(
    standard=StandardScaler,
    minmax=MinMaxScaler,
    maxabs=MaxAbsScaler,
    robust=RobustScaler,
)

# All available encoding strategies
ENCODING_STRATS = dict(
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

# All available pruning strategies
PRUNING_STRATS = dict(
    iforest=IsolationForest,
    ee=EllipticEnvelope,
    lof=LocalOutlierFactor,
    svm=OneClassSVM,
    dbscan=DBSCAN,
    optics=OPTICS,
)

# All available balancing strategies
BALANCING_STRATS = dict(
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
    """Utility to reduce sequences with just one item."""
    return item[0] if isinstance(item, SEQUENCE) and len(item) == 1 else item


def lst(item):
    """Utility used to make sure an item is iterable."""
    return [item] if not isinstance(item, SEQUENCE) else item


def dct(item):
    """Utility used to handle mutable arguments."""
    return {} if item is None else item


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


def check_multidimensional(df):
    """Check if the dataframe contains a multidimensional column."""
    return df.columns[0] == "Multidimensional feature" and len(df.columns) <= 2


def check_method(cls, method):
    """Raise an error if the dataset has more than two dimensions."""
    if check_multidimensional(cls.X):
        raise PermissionError(
            f"The {method} method is not available for "
            f"datasets with more than two dimensions!"
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
    """Check if the data is scaled to mean=0 and std=1."""
    return True if X.mean().mean() < 0.05 and 0.93 < X.std().mean() < 1.07 else False


def get_corpus(X):
    """Get text column from dataframe."""
    try:
        return [col for col in X if col.lower() == "corpus"][0]
    except IndexError:
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

    metric: int, optional (default=0)
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
    time: str
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


def to_df(data, index=None, columns=None, pca=False):
    """Convert a dataset to pd.Dataframe.

    Parameters
    ----------
    data: list, tuple, dict, np.ndarray, pd.DataFrame or None
        Dataset to convert to a dataframe.  If None, return
        unchanged.

    index: sequence or Index
        Values for the dataframe's index.

    columns: sequence or None, optional (default=None)
        Name of the columns. Use None for automatic naming.

    pca: bool, optional (default=False)
        Whether the columns are Features or Components.

    Returns
    -------
    df: pd.DataFrame or None
        Transformed dataframe.

    """
    if data is not None and not isinstance(data, pd.DataFrame):
        if not isinstance(data, dict):  # Dict already has column names
            if columns is None and not pca:
                columns = [f"Feature {str(i)}" for i in range(1, len(data[0]) + 1)]
            elif columns is None:
                columns = [f"Component {str(i)}" for i in range(1, len(data[0]) + 1)]
        data = pd.DataFrame(data, index=index, columns=columns)

    return data


def to_series(data, index=None, name="Target"):
    """Convert a column to pd.Series.

    Parameters
    ----------
    data: sequence or None
        Data to convert. If None, return unchanged.

    index: sequence or Index, optional (default=None)
        Values for the indices.

    name: string, optional (default="Target")
        Name of the target column.

    Returns
    -------
    series: pd.Series or None
        Transformed series.

    """
    if data is not None and not isinstance(data, pd.Series):
        data = pd.Series(data, index=index, name=name)

    return data


def arr(df):
    """From dataframe to multidimensional array.

    When the data consist of more than 2 dimensions, ATOM stores
    it in a df with a single column, "Multidimensional feature".
    This function extracts the arrays from every row and returns
    them stacked.

    Parameters
    ----------
    df: pd.DataFrame
        Dataset to check.

    Returns
    -------
    df: pd.DataFrame
        Stacked dataframe.

    """
    if check_multidimensional(df):
        return np.stack(df["Multidimensional feature"].values)
    else:
        return df


def prepare_logger(logger, class_name):
    """Prepare logging file.

    Parameters
    ----------
    logger: str, class or None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic name.
        - Else: Python `logging.Logger` instance.


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

    for name in MODEL_LIST.keys():
        cond_1 = must_be_equal and model.lower() == name.lower()
        cond_2 = not must_be_equal and model.lower().startswith(name.lower())
        if cond_1 or cond_2:
            return name

    raise ValueError(
        f"Unknown model: {model}! Choose from: {', '.join(MODEL_LIST.keys())}."
    )


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
    acronym:str
        Created acronym.

    """
    from .models import MODEL_LIST

    acronym = "".join([c for c in fullname if c.isupper()])
    if len(acronym) < 2 or acronym.lower() in MODEL_LIST:
        return fullname
    else:
        return acronym


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

    # If it's not any of the predefined models, create a new acronym
    return create_acronym(get_name(estimator)), get_name(estimator)


def get_scorer(metric, gib=True, needs_proba=False, needs_threshold=False):
    """Get a scorer from a str, func or scorer.

    Scorers used by ATOM have a name attribute.

    Parameters
    ----------
    metric: str, func or scorer
        Name, metric or scorer to get ATOM's scorer from.

    gib: bool, optional (default=True)
        whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower is
        better. Is ignored if the metric is a string or a scorer.

    needs_proba: bool, optional (default=False)
        Whether the metric function requires probability estimates of
        a classifier. Is ignored if the metric is a string or a scorer.

    needs_threshold: bool, optional (default=False)
        Whether the metric function takes a continuous decision
        certainty. Is ignored if the metric is a string or a scorer.

    Returns
    -------
    scoring: scorer
        Scorer object with name attribute.

    """

    def get_scorer_name(scorer):
        """Return the name of the provided scorer."""
        for key, value in SCORERS.items():
            if scorer.__dict__ == value.__dict__:
                return key

    if isinstance(metric, str):
        metric = metric.lower()
        if metric in SCORERS:
            scorer = SCORERS[metric]
            scorer.name = metric
        elif metric in SCORERS_ACRONYMS:
            scorer = SCORERS[SCORERS_ACRONYMS[metric]]
            scorer.name = SCORERS_ACRONYMS[metric]
        elif metric in CUSTOM_SCORERS:
            scorer = make_scorer(CUSTOM_SCORERS[metric])
            scorer.name = scorer._score_func.__name__
        else:
            raise ValueError(
                "Unknown value for the metric parameter, got "
                f"{metric}. Choose from: {', '.join(SCORERS)}."
            )

    elif hasattr(metric, "_score_func"):  # Scoring is a scorer
        scorer = metric
        scorer.name = get_scorer_name(scorer)

    else:  # Scoring is a function with signature metric(y, y_pred)
        scorer = make_scorer(
            score_func=metric,
            greater_is_better=gib,
            needs_proba=needs_proba,
            needs_threshold=needs_threshold,
        )
        scorer.name = scorer._score_func.__name__

    return scorer


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
        features (where the partial dependence is evaluated), and
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

def custom_transform(self, transformer, branch, data=None, verbose=None):
    """Applies a transformer on a branch.

    This function is generic and should work for all
    methods with parameters X and/or y.

    Parameters
    ----------
    self: class
        Instance from which the function is called.

    transformer: estimator
        Transformer to apply to the data.

    branch: Branch
        Transformer's branch.

    data: tuple or None
        New data to transform on. If tuple, should have form
        (X, y). If None, the transformation is applied directly
        on the branch.

    verbose: int or None, optional (default=None)
        Verbosity level for the transformation. If None, the
        transformer's verbosity is used.

    """

    def name_cols(array, df):
        """Get the column names after a transformation.

        If the number of columns is unchanged, the original
        column names are returned. Else, give the column a
        default name if the column values changed.

        Parameters
        ----------
        array: np.ndarray
            Transformed dataset.

        df: pd.DataFrame
            Original dataset.

        """
        # If columns were only transformed, return og names
        if array.shape[1] == df.shape[1]:
            return df.columns

        # If columns were added or removed
        temp_cols = []
        for i, col in enumerate(array.T, start=1):
            mask = df.apply(lambda c: all(c == col))
            if any(mask):
                temp_cols.append(mask[mask].index.values[0])
            else:
                diff = branch.n_features - len(transformer.cols)
                temp_cols.append(f"Feature {str(i + diff)}")

        return temp_cols

    def reorder_cols(df, original_df):
        """Reorder the columns to their original order.

        This function is necessary in case only a subset of the
        columns in the dataset was used. In that case, we need
        to reorder them to their original order.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame to reorder.

        original_df: pd.DataFrame
            Original dataframe (states the order).

        """
        temp_df = pd.DataFrame()
        for col in list(dict.fromkeys(list(original_df.columns) + list(df.columns))):
            if col in df.columns:
                temp_df[col] = df[col]
            elif col not in transformer.cols:
                temp_df[col] = original_df[col]

            # Derivative cols are added after original
            for col_derivative in df.columns:
                if col_derivative.startswith(col):
                    temp_df[col_derivative] = df[col_derivative]

        return temp_df

    # Select provided data or from the branch
    if data:
        X_og, y_og = to_df(data[0]), to_series(data[1])
    else:
        if transformer.train_only:
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

    if transformer.__class__.__name__ in ("DropTransformer", "FuncTransformer"):
        X, y = transformer.transform(X_og, y_og)
    else:
        if not transformer.__module__.startswith("atom"):
            self.log(f"Applying {transformer.__class__.__name__} to the dataset...", 1)

        args = []
        if "X" in signature(transformer.transform).parameters:
            args.append(X_og[transformer.cols])
        if "y" in signature(transformer.transform).parameters:
            args.append(y_og)
        output = transformer.transform(*args)

        # Transform can return X, y or both
        if isinstance(output, tuple):
            X, y = output[0], output[1]
        else:
            if len(output.shape) > 1:
                X, y = output, y_og  # Only X
            else:
                X, y = X_og, output  # Only y

        # Convert to pandas and assign proper column names
        if not isinstance(X, pd.DataFrame):
            X = to_df(X, columns=name_cols(X, X_og[transformer.cols]))
        X = reorder_cols(X, X_og)
        y = to_series(y, name=branch.target)

    # Apply changes to the branch
    if not data:
        if transformer.train_only:
            branch.train = merge(X, branch.y_train if y is None else y)
        else:
            branch.dataset = merge(X, branch.y if y is None else y)

            # Since rows can be removed from train and test, reset indices
            branch.idx[1] = len(X[X.index >= branch.idx[0]])
            branch.idx[0] = len(X[X.index < branch.idx[0]])

        branch.dataset = branch.dataset.reset_index(drop=True)

    # Back to the original verbosity
    if verbose is not None and hasattr(transformer, "verbose"):
        transformer.verbose = vb

    return X, y


def add_transformer(self, transformer, columns=None, train_only=False, **fit_params):
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

    transformer: class instance
        Transformer to add to the pipeline. Should implement a
        `transform` method.

    columns: int, str, slice, sequence or None, optional (default=None)
        Names or indices of the columns in the dataset to transform.
        If None, transform all columns.

    train_only: bool, optional (default=False)
        Whether to apply the transformer only on the train set or
        on the complete dataset.

    **fit_params
        Additional keyword arguments passed to the transformer's fit
        method.

    """
    if not hasattr(transformer, "transform"):
        raise ValueError("Added transformers should have a transform method!")

    # Add BaseTransformer params to the estimator if left to default
    if all(hasattr(transformer, attr) for attr in ("get_params", "set_params")):
        sign = signature(transformer.__init__).parameters
        for p in ("n_jobs", "random_state"):
            if p in sign and transformer.get_params()[p] == sign[p]._default:
                transformer.set_params(**{p: getattr(self, p)})

    # Transformers remember the train_only and columns parameters
    transformer.train_only = train_only
    transformer.cols = [col for col in self._get_columns(columns) if col != self.target]

    if hasattr(transformer, "fit") and not check_is_fitted(transformer, False):
        if not transformer.__module__.startswith("atom"):
            self.log(f"Fitting {transformer.__class__.__name__}...", 1)

        args = []
        if "X" in signature(transformer.fit).parameters:
            args.append(self.X_train[transformer.cols])
        if "y" in signature(transformer.fit).parameters:
            args.append(self.y_train)
        transformer.fit(*args, **fit_params)

    custom_transform(self, transformer, self.branch, verbose=self.verbose)

    # Add the transformer to the pipeline
    self.branch.pipeline = self.pipeline.append(
        pd.Series([transformer], name=self._current), ignore_index=True
    )


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


def score_decorator(f):
    """Decorator for sklearn's _score function.

    Special `hack` for sklearn.model_selection._validation._score
    in order to score pipelines that drop samples during transforming.

    """

    def wrapper(*args, **kwargs):
        args = list(args)  # Convert to list for item assignment
        if len(args[0]) > 1:  # Has transformers
            args[1], args[2] = args[0][:-1].transform(args[1], args[2])

        # Return f(final_estimator, X_transformed, y_transformed, ...)
        return f(args[0][-1], *tuple(args[1:]), **kwargs)

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


def lift(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float((tp / (tp + fp)) / ((tp + fn) / (tp + tn + fp + fn)))


# Scorers not predefined by sklearn
CUSTOM_SCORERS = dict(
    tn=true_negatives,
    fp=false_positives,
    fn=false_negatives,
    tp=true_positives,
    fpr=false_positive_rate,
    tpr=true_positive_rate,
    tnr=true_negative_rate,
    fnr=false_negative_rate,
    lift=lift,
    mcc=matthews_corrcoef,
)


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


class CustomDict(MutableMapping):
    """Custom ordered dictionary.

    Custom dictionary for the _models and _metric private attributes
    of the trainers. the main differences with the Python dictionary
    are:
        - It has ordered entries.
        - It allows getting an item from an index position.
        - It can insert key value pairs at a specific position.
        - Key requests are case insensitive.
        - If iterated over, it iterates over its values, not the keys!

    """

    @staticmethod
    def _conv(key):
        return key.lower() if isinstance(key, str) else key

    def _get_key(self, key):
        return [k for k in self.__keys if self._conv(k) == self._conv(key)][0]

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
                self.__keys.append(key)
                self.__data[self._conv(key)] = value

        for key, value in kwargs.items():
            self.__keys.append(key)
            self.__data[self._conv(key)] = value

    def __getitem__(self, key):
        try:
            return self.__data[self._conv(key)]  # From key
        except KeyError as e:
            try:
                return self.__data[self._conv(self.__keys[key])]  # From index
            except (TypeError, IndexError):
                raise e

    def __setitem__(self, key, value):
        self.__keys.append(key)
        self.__data[self._conv(key)] = value

    def __delitem__(self, key):
        del self.__data[self._conv(key)]
        self.__keys.remove(self._get_key(key))

    def __iter__(self):
        yield from self.values()

    def __len__(self):
        return len(self.__keys)

    def __contains__(self, key):
        try:
            self._get_key(key)
            return True
        except (AttributeError, IndexError):
            return False

    def __repr__(self):
        return str(dict(self))

    def keys(self):
        yield from self.__keys

    def items(self):
        for key in self.__keys:
            yield key, self.__data[self._conv(key)]

    def values(self):
        for key in self.__keys:
            yield self.__data[self._conv(key)]

    def insert(self, pos, new_key, value):
        try:
            self.__keys.insert(self.__keys.index(self._get_key(pos)), new_key)
        except (IndexError, KeyError):
            self.__keys.insert(pos, new_key)
        finally:
            self.__data[self._conv(new_key)] = value

    def get(self, key, default=None):
        try:
            return self[key]
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
            return self.__data.pop(self._conv(self.__keys.pop()))
        except IndexError:
            raise KeyError(f"{self.__class__.__name__} is empty")

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
                if key not in self:
                    self.__keys.append(key)
                self.__data[self._conv(key)] = value

        for key, value in kwargs.items():
            if key not in self:
                self.__keys.append(key)
            self.__data[self._conv(key)] = value

    def setdefault(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return self[key]

    def index(self, key):
        try:
            return self.__keys.index(self._get_key(key))
        except IndexError:
            raise KeyError(key)

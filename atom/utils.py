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
from sklearn.metrics import SCORERS, get_scorer, make_scorer

# Plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Global constants ========================================================== >>

# Variable types
CAL = Union[str, callable]
X_TYPES = Union[dict, Sequence[Sequence], np.ndarray, pd.DataFrame]
Y_TYPES = Union[int, str, list, tuple, dict, np.ndarray, pd.Series]
TRAIN_TYPES = Union[Sequence[Union[int, float]], np.ndarray]

# Tuple of models that need to import an extra package
OPTIONAL_PACKAGES = (('XGB', 'xgboost'),
                     ('LGB', 'lightgbm'),
                     ('CatB', 'catboost'))

# List of models that only work for regression/classification tasks
ONLY_CLASSIFICATION = ['BNB', 'GNB', 'MNB', 'LR', 'LDA', 'QDA']
ONLY_REGRESSION = ['OLS', 'Lasso', 'EN', 'BR']

# List of tree-based models
TREE_MODELS = ['Tree', 'Bag', 'ET', 'RF', 'AdaB', 'GBM', 'XGB', 'LGB', 'CatB']

# List of all the model acronyms
MODEL_NAMES = ['GP', 'GNB', 'MNB', 'BNB', 'OLS', 'Ridge', 'Lasso', 'EN', 'BR',
               'LR', 'LDA', 'QDA', 'KNN', 'Tree', 'Bag', 'ET', 'RF', 'AdaB', 'GBM',
               'XGB', 'LGB', 'CatB', 'lSVM', 'kSVM', 'PA', 'SGD', 'MLP']

METRIC_ACRONYMS = dict(ap='average_precision',
                       AP='average_precision',
                       ba='balanced_accuracy',
                       BA='balanced_accuracy',
                       auc='roc_auc',
                       AUC='roc_auc',
                       ev='explained_variance',
                       EV='explained_variance',
                       me='max_error',
                       ME='max_error',
                       mae='neg_mean_absolute_error',
                       MAE='neg_mean_absolute_error',
                       mse='neg_mean_squared_error',
                       MSE='neg_mean_squared_error',
                       rmse='neg_root_mean_squared_error',
                       RMSE='neg_root_mean_squared_error',
                       msle='neg_mean_squared_log_error',
                       MSLE='neg_mean_squared_log_error',
                       medae='neg_median_absolute_error',
                       MEDAE='neg_median_absolute_error',
                       poisson='neg_mean_poisson_deviance',
                       POISSON='neg_mean_poisson_deviance',
                       gamma='neg_mean_gamma_deviance',
                       GAMMA='neg_mean_gamma_deviance')


# Functions ================================================================= >>

def flt(item):
    """Return value if item is list of length 1."""
    return item[0] if isinstance(item, list) and len(item) == 1 else item


def lst(item):
    """Return list if item is not a list."""
    return [item] if not isinstance(item, list) else item


def merge(X, y):
    """Merge a pd.DataFrame and pd.Series into one dataframe."""
    return X.merge(y.to_frame(), left_index=True, right_index=True)


def catch_return(args):
    """Returns always two arguments independent of length args."""
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


def check_scaling(X):
    """Check if the provided data is scaled to mean=0 and std=1."""
    mean = X.mean(axis=1).mean()
    std = X.std(axis=1).mean()
    return True if mean < 0.05 and 0.5 < std < 1.5 else False


def get_best_score(item, metric=0):
    """Returns the bagging or test score of a model.

    Parameters
    ----------
    item: model subclass or pd.Series
        Model subclass instance or row from the results dataframe.

    metric: int, optional (default=0)
        Index of the metric to use.

    """
    if item.mean_bagging:
        return lst(item.mean_bagging)[metric]
    else:
        return lst(item.score_test)[metric]


def time_to_string(t_init):
    """Convert time integer to string.

    Convert a time duration to a string of format 00h:00m:00s
    or 1.000s if under 1 min.

    Parameters
    ----------
    t_init: float
        Time to convert (in seconds).

    """
    t = time() - t_init  # Total time in seconds
    h = int(t/3600.)
    m = int(t/60.) - h*60
    s = t - h*3600 - m*60
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
    data: array-like
        Dataset to convert to a dataframe.

    index: array-like or Index
        Values for the dataframe's index.

    columns: array-like or None, optional(default=None)
        Name of the columns in the dataset. If None, the names are autofilled.

    pca: bool, optional (default=False)
        whether the columns need to be called Features or Components.

    Returns
    -------
    df: pd.DataFrame
        Transformed dataframe.

    """
    if isinstance(data, pd.DataFrame):
        return data
    else:
        if columns is None and not pca:
            columns = ['Feature ' + str(i) for i in range(len(data[0]))]
        elif columns is None:
            columns = ['Component ' + str(i) for i in range(len(data[0]))]
        return pd.DataFrame(data, index=index, columns=columns)


def to_series(data, index=None, name='target'):
    """Convert a column to pd.Series.

    Parameters
    ----------
    data: list, tuple or np.array
        Data to convert.

    index: array or Index, optional (default=None)
        Values for the series' index.

    name: string, optional (default='target')
        Name of the target column.

    Returns
    -------
    series: pd.Series
        Transformed series.

    """
    if isinstance(data, pd.Series):
        return data
    else:
        return pd.Series(data, index=index, name=name)


def prepare_logger(logger, class_name):
    """Prepare logging file.

    Parameters
    ----------
    logger: bool, str, class or None
        - If None: No logger.
        - If bool: True for logging file with default name, False for no logger.
        - If string: name of the logging file. 'auto' for default name.
        - If class: python Logger object.

        The default name created by ATOM contains the class name followed by the
        timestamp  of the logger's creation, e.g. ATOMClassifier_

    class_name: str
        Name of the class from which the function is called.
        Used for default name creation when log='auto'.

    Returns
    -------
    logger: class
        Logger object.

    """
    if logger is True:
        logger = 'auto'

    if not logger:  # Empty string, False or None
        return

    elif isinstance(logger, str):
        # Prepare the FileHandler's name
        if not logger.endswith('.log'):
            logger += '.log'
        if logger == 'auto.log' or logger.endswith('/auto.log'):
            current = datetime.now().strftime("%d%b%y_%Hh%Mm%Ss")
            logger = logger.replace('auto', class_name + '_logger_' + current)

        # Define file handler and set formatter
        file_handler = logging.FileHandler(logger)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        file_handler.setFormatter(formatter)

        # Define logger
        logger = logging.getLogger(class_name + '_logger')
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if logger.hasHandlers():  # Remove existing handlers
            logger.handlers.clear()
        logger.addHandler(file_handler)  # Add file handler to logger

    elif type(logger) != logging.Logger:  # Should be python Logger object
        raise TypeError("Invalid value for the logger parameter. Should be a " +
                        f"python logging.Logger object, got {type(logger)}!")

    return logger


def check_property(value, value_name,
                   side=None, side_name=None,
                   under=None, under_name=None):
    """Check the property setter on type and dimensions.

    Convert the property to a pandas object and compare with other
    properties to check if it has the right dimensions.

    Parameters
    ----------
    value: sequence, np.array or pd.DataFrame/pd.Series
        Property to be checked.

    value_name: str
        Name of the property to check.

    side: pd.DataFrame/pd.Series or None, optional (default=None)
        Other property to compare the length with.

    side_name: str
        Name of the property of the side parameter.

    under: pd.DataFrame/pd.Series or None, optional (default=None)
        Other property to compare the width with.

    under_name: str
        Name of the property of the under parameter.

    """
    index = side.index if side_name else None
    if 'y' in value_name:
        name = under.name if under_name else 'target'
        value = to_series(value, index=index, name=name)
    else:
        columns = under.columns if under_name else None
        value = to_df(value, index=index, columns=columns)

    if side_name:  # Check for equal number of rows
        if len(value) != len(side):
            raise ValueError(
                f"The {value_name} and {side_name} properties need to have the " +
                f"same number of rows, got {len(value)} != {len(side)}.")
        if not value.index.equals(side.index):
            raise ValueError(
                f"The {value_name} and {side_name} properties need to have the " +
                f"same indices, got {value.index} != {side.index}.")

    if under_name:  # Check they have the same columns
        if 'y' in value_name:
            if value.name != under.name:
                raise ValueError(
                    f"The {value_name} and {under_name} properties need to have " +
                    f"the same name, got {value.name} != {under.name}.")
        else:
            if value.shape[1] != under.shape[1]:
                raise ValueError(
                    f"The {value_name} and {under_name} properties need to have " +
                    f"the same number of columns, got {value.shape[1]} != " +
                    f"{under.shape[1]}.")

            if list(value.columns) != list(under.columns):
                raise ValueError(
                    f"The {value_name} and {under_name} properties need to have " +
                    f"the same columns , got {value.columns} != {under.columns}.")

    return value


def check_is_fitted(estimator, attributes=None, msg=None):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of fitted
    attributes (not None or empty) and otherwise raises a NotFittedError.
    This is a variation on sklearn's function.

    Parameters
    ----------
    estimator: class
        Class instance for which the check is performed.

    attributes: str, sequence or None, optional (default=None)
        Attribute value_name(s) to check. If None, estimator is considered fitted if
        there exist an attribute that ends with a underscore and does not
        start with double underscore.

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
        msg = (f"This {type(estimator).__name__} instance is not fitted " +
               "yet. Call 'fit' with appropriate arguments before using " +
               "this estimator.")

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = [check_attr(attr) for attr in attributes]
    else:
        attrs = []
        for key, value in vars(estimator).items():
            if key.endswith("_") and not key.startswith("__"):
                attrs.append(not value)

    if all(attrs):
        raise NotFittedError(msg)


def get_model_name(model):
    """Get the right model acronym.

    Parameters
    ----------
    model: str
        Acronym of the model, case insensitive.

    Returns
    -------
    name: str
        Correct model name acronym as present in the MODEL_LIST constant.

    """
    # Compare strings case insensitive
    if model.lower() not in map(str.lower, MODEL_NAMES):
        raise ValueError(
            f"Unknown model: {model}! Choose from: {', '.join(MODEL_NAMES)}.")
    else:
        for name in MODEL_NAMES:
            if model.lower() == name.lower():
                return name


def get_metric(metric, greater_is_better, needs_proba, needs_threshold):
    """Get the right metric depending on input type.

    Parameters
    ----------
    metric: str, callable or None
        Metric as a string, function or scorer.

    greater_is_better: bool
        whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower is
        better. Will be ignored if the metric is a string or a scorer.

    needs_proba: bool
        Whether the metric function requires probability estimates out of a
        classifier. Will be ignored if the metric is a string or a scorer.

    needs_threshold: bool
        Whether the metric function takes a continuous decision certainty.
        Will be ignored if the metric is a string or a scorer.

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

    if metric is None:
        return
    elif isinstance(metric, str):
        if metric in METRIC_ACRONYMS:
            metric = METRIC_ACRONYMS[metric]
        elif metric not in SCORERS:
            raise ValueError("Unknown value for the metric parameter, got " +
                             f"{metric}. Try one of: {', '.join(SCORERS)}.")
        metric = get_scorer(metric)
        metric.name = get_scorer_name(metric)

    elif hasattr(metric, '_score_func'):  # Provided metric is scoring
        metric.name = get_scorer_name(metric)

    else:  # Metric is a function with signature metric(y, y_pred)
        metric = make_scorer(metric,
                             greater_is_better=greater_is_better,
                             needs_proba=needs_proba,
                             needs_threshold=needs_threshold)
        metric.name = metric._score_func.__name__

    return metric


def get_default_metric(task):
    """Return the default metric for each task.

    Parameters
    ----------
    task: str
        One of binary classification, multiclass classification or regression.

    """
    if task.startswith('bin'):
        return get_metric('f1', True, False, False)
    elif task.startswith('multi'):
        return get_metric('f1_weighted', True, False, False)
    else:
        return get_metric('r2', True, False, False)


def infer_task(y, goal=''):
    """Infer the task corresponding to a target column.

    If goal is provided, only look at number of unique values to determine the
    classification task. If not, returns binary for 2 unique values, multiclass
    if the number of unique values in y is <10% of the values and values>100,
    else returns regression.

    Parameters
    ----------
    y: pd.Series
        Target column from which to infer the task.

    goal: str, optional (default='')
        Classification or regression goal. Empty to infer the task from
        the number of unique values in y.

    Returns
    -------
    task: str
        Inferred task.

    """
    unique = y.unique()
    if len(unique) == 1:
        raise ValueError(f"Only found 1 target value: {unique[0]}")

    if goal.startswith('reg'):
        return 'regression'

    if goal.startswith('class'):
        if len(unique) == 2:
            return 'binary classification'
        else:
            return 'multiclass classification'

    if not goal:
        if len(unique) == 2:
            return 'binary classification'
        elif len(unique) < 0.1 * len(y) and len(unique) < 30:
            return 'multiclass classification'
        else:
            return 'regression'


# Functions shared by classes =============================================== >>

def clear(self, models):
    """Clear models from the trainer.

    If the winning model is removed. The next best model (through
    score_test or mean_bagging if available) is selected as winner.

    Parameters
    ----------
    self: class
        Class for which to clear the model.

    models: sequence
        Name of the models to clear from the pipeline.

    """
    for model in models:
        if model not in self.models:
            raise ValueError(f"Model {model} not found in pipeline!")
        else:
            self.models.remove(model)
            if isinstance(self._results.index, pd.MultiIndex):
                self._results = self._results.iloc[
                    ~self._results.index.get_level_values(1).str.contains(model)]
            else:
                self._results.drop(model, axis=0, inplace=True, errors='ignore')

            if not self.models:  # No more models in the pipeline
                self.metric = None

            # Delete model subclasses
            delattr(self, model)
            delattr(self, model.lower())


# Decorators ================================================================ >>

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


def crash(f, cache={'last_exception': None}):
    """Save program crashes to log file.

    We use a mutable argument to cache the last exception raised. If the current
    exception is the same (happens when there is an error catch or multiple calls
    to crash), its not written again to the logger.

    """
    def wrapper(*args, **kwargs):
        logger = args[0].T.logger if hasattr(args[0], 'T') else args[0].logger

        if logger is not None:
            try:  # Run the function
                return f(*args, **kwargs)

            except Exception as exception:
                # If exception is not same as last, write to log
                if exception is not cache['last_exception']:
                    cache['last_exception'] = exception
                    logger.exception("Exception encountered:")

                raise exception  # Always raise it
        else:
            return f(*args, **kwargs)

    return wrapper


def method_to_log(f):
    """Save function's Parameters to log file."""
    def wrapper(*args, **kwargs):
        # Get logger (for model subclasses called from BasePredictor)
        logger = args[0].T.logger if hasattr(args[0], 'T') else args[0].logger

        if logger is not None:
            # For the __init__ method, call extra arguments from api.py
            if f.__name__ != '__init__':
                logger.info('')
            logger.info(f"{args[0].__class__.__name__}.{f.__name__}()")

        result = f(*args, **kwargs)
        return result

    return wrapper


def plot_from_model(f):
    """If a plot is called from a model, adapt the models parameter."""
    def wrapper(*args, **kwargs):
        if hasattr(args[0], 'T'):
            result = f(args[0].T, args[0].name, *args[1:], **kwargs)
        else:
            result = f(*args, **kwargs)
        return result

    return wrapper


# Classes =================================================================== >>

class NotFittedError(ValueError, AttributeError):
    """Exception called when the instance is not yet fitted."""
    pass


class PlotCallback(object):
    """Callback to plot the BO's progress as it runs."""

    def __init__(self, model):
        """Initialize class.

        Parameters
        ----------
        model: class
            Model subclass.

        """
        self.M = model

        # Plot attributes
        max_len = 15  # Maximum steps to show at once in the plot
        self.x = deque(list(range(1, max_len+1)), maxlen=max_len)
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
            self.line1, self.line2, self.ax1, self.ax2 = self.create_canvas()
        self.animate_plot()

    def create_canvas(self):
        """Create the plot.

        Creates a canvas with two plots. The first plot shows the score of
        every trial and the second shows the distance between the last
        consecutive steps.

        """
        plt.ion()  # Call to matplotlib that allows dynamic plotting

        # Initialize plot
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 1, height_ratios=[2, 1])

        # First subplot (without xtick labels)
        ax1 = plt.subplot(gs[0])
        # Create a variable for the line so we can later update it
        line1, = ax1.plot(self.x, self.y1, '-o', alpha=0.8)
        ax1.set_title(f"Bayesian Optimization for {self.M.longname}",
                      fontsize=self.M.T.title_fontsize)
        ax1.set_ylabel(self.M.T.metric[0].name,
                       fontsize=self.M.T.label_fontsize,
                       labelpad=12)
        ax1.set_xlim(min(self.x)-0.5, max(self.x)+0.5)

        # Second subplot
        ax2 = plt.subplot(gs[1], sharex=ax1)
        line2, = ax2.plot(self.x, self.y2, '-o', alpha=0.8)
        ax2.set_title("Distance between last consecutive iterations",
                      fontsize=self.M.T.title_fontsize)
        ax2.set_xlabel('Iteration',
                       fontsize=self.M.T.label_fontsize,
                       labelpad=12)
        ax2.set_ylabel('d',
                       fontsize=self.M.T.label_fontsize,
                       labelpad=12)
        ax2.set_xticks(self.x)
        ax2.set_xlim(min(self.x)-0.5, max(self.x)+0.5)
        ax2.set_ylim([-0.05, 0.1])

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        plt.xticks(fontsize=self.M.T.tick_fontsize)
        plt.yticks(fontsize=self.M.T.tick_fontsize)
        fig.tight_layout()
        plt.show()

        return line1, line2, ax1, ax2

    def animate_plot(self):
        """Plot the BO's progress as it runs."""
        self.line1.set_xdata(self.x)
        self.line1.set_ydata(self.y1)
        self.line2.set_xdata(self.x)
        self.line2.set_ydata(self.y2)
        self.ax1.set_xlim(min(self.x)-0.5, max(self.x)+0.5)
        self.ax2.set_xlim(min(self.x)-0.5, max(self.x)+0.5)
        self.ax1.set_xticks(self.x)
        self.ax2.set_xticks(self.x)

        # Adjust y limits if new data goes beyond bounds
        lim = self.line1.axes.get_ylim()
        if np.nanmin(self.y1) <= lim[0] or np.nanmax(self.y1) >= lim[1]:
            self.ax1.set_ylim([np.nanmin(self.y1) - np.nanstd(self.y1),
                               np.nanmax(self.y1) + np.nanstd(self.y1)])
        lim = self.line2.axes.get_ylim()
        if np.nanmax(self.y2) >= lim[1]:
            self.ax2.set_ylim([-0.05, np.nanmax(self.y2) + np.nanstd(self.y2)])

        # Pause the data so the figure/axis can catch up
        plt.pause(0.01)

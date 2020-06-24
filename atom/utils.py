# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing utility constants, functions and classes.

"""

# Standard packages
import math
import pickle
import logging
import inspect
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


# Functions ================================================================= >>

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


def get_best_score(row):
    """Returns the bagging score or test score of a model.

    Parameters
    ----------
    row: pd.Series
        A row of the results dataframe.

    """
    if 'mean_bagging' in row and row.mean_bagging and not np.isnan(row.mean_bagging):
        return row.mean_bagging
    else:
        return row.score_test


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
        return f'{s:.3f}s'
    elif h < 1:  # Also minutes
        return f'{m}m:{int(s):02}s'
    else:  # Also hours
        return f'{h}h:{m:02}m:{int(s):02}s'


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
    if msg is None:
        msg = (f"This {type(estimator).__name__} instance is not fitted " +
               "yet. Call 'fit' with appropriate arguments before using " +
               "this estimator.")

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = [not getattr(estimator, attr) for attr in attributes]
    else:
        attrs = []
        for key, value in vars(estimator).items():
            if key.endswith("_") and not key.startswith("__"):
                attrs.append(not value)

    if all(attrs):
        raise NotFittedError(msg)


def get_metric(metric, greater_is_better, needs_proba, needs_threshold):
    """Get the right metric depending on input type.

    Parameters
    ----------
    metric: str or callable
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
    if isinstance(metric, str):
        if metric not in SCORERS:
            raise ValueError("Unknown value for the metric parameter, got " +
                             f"{metric}. Try one of: {', '.join(SCORERS)}.")
        return get_scorer(metric)

    elif hasattr(metric, '_score_func'):  # Provided metric is scoring
        return metric

    else:  # Metric is a function with signature metric(y, y_pred)
        return make_scorer(metric, greater_is_better,
                           needs_proba, needs_threshold)


def get_metric_name(scorer):
    """Get the metric name given a scorer.

    Assign the key of the SCORERS constant which value corresponds with the
    provided scorer. If none matches, assign the function's name.

    Parameters
    ----------
    scorer: callable
        SCorer object from which we want to retrieve the name.

    Returns
    -------
    name: str
        Name of the metric.

    """
    for key, value in SCORERS.items():
        if scorer.__dict__ == value.__dict__:
            return key

    return scorer._score_func.__name__


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


def get_train_test(*arrays):
    """Converts a list of input data into a train and test set.

    Parameters
    ----------
    arrays: array-like
        Either a train and test set or X_train, X_test, y_train, y_test.

    Returns
    -------
    train: pd.DataFrame
        Training set for the BaseTrainer.

    test: pd.DataFrame
        Test set for the BaseTrainer.

    """
    if len(arrays) == 2:
        train, test = to_df(arrays[0]), to_df(arrays[1])
    elif len(arrays) == 4:
        train = merge(to_df(arrays[0]), to_series(arrays[2]))
        test = merge(to_df(arrays[1]), to_series(arrays[3]))
    else:
        raise ValueError(
            "Invalid parameters. Must be either of the form (train, " +
            "test) or (X_train, X_test, y_train, y_test).")

    return train, test


def attach_methods(self, class_, names):
    """Attach methods from one class to the other.

    Search for methods in class_ and all parent classes of class_ whose name
    start with names and attach them to self.

    self: class
        Class to attach methods.

    class_: class
        Class from which the methods come.

    names: str or sequence
        Names with which the methods to transfer start.

    """
    if isinstance(names, str):
        names = [names]

    items = dict(class_.__dict__)  # __dict__ has mappingproxy type
    # Add all parent's methods to the items dictionary
    for parent in inspect.getmro(class_):
        items.update(**parent.__dict__)

    for key, value in items.items():
        if any([key.startswith(name) for name in names]):
            setattr(self, key, getattr(class_, key).__get__(self))


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
                self.winner = None
                self.metric = None

            else:
                # Assign new winning model (if it changed)
                if self.winner is getattr(self, model, ''):
                    # Make a df of the scores for all remaining models
                    scores = pd.DataFrame(columns=['score_test', 'mean_bagging'])
                    for m in self.models:
                        scores.loc[m] = [getattr(self, m).score_test,
                                         getattr(self, m).mean_bagging]

                    best = scores.apply(lambda row: get_best_score(row), axis=1)
                    self.winner = getattr(self, str(best.idxmax()))

            # Delete model subclasses
            delattr(self, model)
            delattr(self, model.lower())


def save(self, filename):
    """Save class to a pickle file.

    Parameters
    ----------
    self: class
        Class from which the function is called.

    filename: str or None, optional (default=None)
        Name of the file when saved (as .html). None to not save anything.

    """
    filename = filename if filename.endswith('.pkl') else filename + '.pkl'
    pickle.dump(self, open(filename, 'wb'))


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


def crash(f):
    """Save program crashes to log file."""
    def wrapper(*args, **kwargs):
        logger = args[0].T.logger if hasattr(args[0], 'T') else args[0].logger

        if logger is not None:
            try:  # Run the function
                return f(*args, **kwargs)

            except Exception as exception:
                # Write exception to log and raise it
                logger.exception("Exception encountered:")
                raise eval(type(exception).__name__)(exception)
        else:
            return f(*args, **kwargs)

    return wrapper


def params_to_log(f):
    """Save function's Parameters to log file."""
    def wrapper(*args, **kwargs):
        logger = args[0].T.logger if hasattr(args[0], 'T') else args[0].logger
        kwargs_list = ['{}={!r}'.format(k, v) for k, v in kwargs.items()]
        args_list = [str(i) for i in args[1:]]
        args_list = args_list + [''] if len(kwargs_list) != 0 else args_list

        if logger is not None:
            # For the __init__ method, call extra arguments from api.py
            if f.__name__ == '__init__':
                extra_kwargs = dict(n_jobs=args[0].n_jobs,
                                    warnings=args[0].warnings,
                                    verbose=args[0].verbose,
                                    logger=args[0].logger.handlers[-1].baseFilename,
                                    random_state=args[0].random_state)
                kwargs_2 = ['{}={!r}'.format(k, v) for k, v in extra_kwargs.items()]
                logger.info(f"{args[0].__class__.__name__}.{f.__name__}" +
                            f"(X, y, {', '.join(kwargs_list + kwargs_2)})")
            else:
                logger.info('')  # Empty line first
                logger.info(f"{args[0].__class__.__name__}.{f.__name__}" +
                            f"({', '.join(args_list)}{', '.join(kwargs_list)})")

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
        ax1.set_ylabel(self.M.T.metric.name,
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

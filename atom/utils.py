# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Module containing utility functions.

"""


# << ============ Import Packages ============ >>

# Standard packages
import logging
import pandas as pd
from time import time
from datetime import datetime


# << ============ Functions ============ >>

def prepare_logger(log):

    """
    Prepare logging file.

    Parameters
    ----------
    log: string
        Name of the logging file.

    Returns
    -------
    logger: callable
        Logger object.

    """

    if log == 'auto':
        current = datetime.now().strftime("%d%b%y_%Hh%Mm%Ss")
        logger = 'ATOM_logger_' + current + '.log'
    elif log is None or log.endswith('.log'):
        logger = log
    else:
        logger = log + '.log'

    # Creating logging handler
    if logger is not None:
        # Define file handler and set formatter
        file_handler = logging.FileHandler(logger)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        file_handler.setFormatter(formatter)

        # Define logger
        logger = logging.getLogger('ATOM_logger')
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if logger.hasHandlers():  # Remove existing handlers
            logger.handlers.clear()
        logger.addHandler(file_handler)  # Add file handler to logger

    return logger


def time_to_string(t_init):

    """
    Convert a time duration to a neat string in format 00h:00m:00s
    or 1.000s if under 1 min.

    Parameters
    ----------
    t_init: int
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


def to_df(data, columns=None, pca=False):

    """
    Convert a dataset to pd.Dataframe.

    PARAMETERS
    ----------

    data: list, tuple or np.array
        Dataset to convert to a dataframe.

    columns: list, tuple or None
        Name of the columns in the dataset. If None, the names are autofilled.

    pca: bool
        Wether the columns need to be called Features or Components.

    """

    if columns is None and not pca:
        columns = ['Feature ' + str(i) for i in range(len(data[0]))]
    elif columns is None:
        columns = ['Component ' + str(i) for i in range(len(data[0]))]
    return pd.DataFrame(data, columns=columns)


def to_series(data, name=None):

    """
    Convert data to pd.Series.

    PARAMETERS
    ----------
    data: list, tuple or np.array
        Data to convert.

    name: string or None
        Name of the target column. If None, the name is set to 'target'.

    """

    return pd.Series(data, name=name if name is not None else 'target')


def merge(X, y):
    """ Merge pd.DataFrame and pd.Series into one df """

    return X.merge(y.to_frame(), left_index=True, right_index=True)


def check_is_fitted(is_fitted):
    if not is_fitted:
        raise AttributeError('Run the pipeline before calling this method!')


# << ============ Decorators ============ >>

def composed(*decs):

    """
    Add multiple decorators in one line.

    Parameters
    ----------
    decs: tuple
        Decorators to run.

    """

    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco


def crash(f):
    """ Decorator to save program crashes to log file """

    def wrapper(*args, **kwargs):
        try:  # Run the function
            result = f(*args, **kwargs)
            return result

        except Exception as exception:
            log = args[0].T.log if hasattr(args[0], 'T') else args[0].log

            # Write exception to log and raise it
            if type(log) == logging.Logger:
                log.exception("Exception encountered:")
            raise eval(type(exception).__name__)(exception)

    return wrapper


def params_to_log(f):
    """ Decorator to save function's parameters to log file """

    def wrapper(*args, **kwargs):
        log = args[0].T.log if hasattr(args[0], 'T') else args[0].log
        kwargs_list = ['{}={!r}'.format(k, v) for k, v in kwargs.items()]
        args_list = [str(i) for i in args[1:]]
        args_list = args_list + [''] if len(kwargs_list) != 0 else args_list
        if log is not None:
            if f.__name__ == '__init__':
                log.info(f"Method: {args[0].__class__.__name__}.{f.__name__}" +
                         f"(X, y, {', '.join(kwargs_list)})")
            else:
                log.info('')  # Empty line first
                log.info(f"Method: {args[0].__class__.__name__}.{f.__name__}" +
                         f"({', '.join(args_list)}{', '.join(kwargs_list)})")

        result = f(*args, **kwargs)
        return result

    return wrapper

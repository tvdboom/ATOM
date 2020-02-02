# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Module containing utility functions.

'''


# << ============ Import Packages ============ >>

# Standard packages
import logging
import pandas as pd
from time import time


# << ============ Functions ============ >>

def prepare_logger(log):
    ''' Prepare logging file '''

    if not isinstance(log, (type(None), str)):
        raise_TypeError('log', log)
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


def prlog(string, class_, level=0):

    '''
    DESCRIPTION -----------------------------------

    Print and save output to log file.

    PARAMETERS -------------------------------------

    string --> string to output
    class_ --> class of the element
    level  --> minimum verbosity level to print

    '''

    verbose = class_.T.verbose if hasattr(class_, 'T') else class_.verbose
    if verbose > level:
        print(string)

    log = class_.T.log if hasattr(class_, 'T') else class_.log
    if log is not None:
        while string.startswith('\n'):  # Insert empty lines for clean view
            log.info('')
            string = string[1:]
        log.info(string)


def time_to_string(t_init):
    ''' Returns time duration as a neat string '''

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

    '''
    DESCRIPTION -----------------------------------

    Convert data to pd.Dataframe.

    PARAMETERS -------------------------------------

    data    --> dataset to convert
    columns --> name of the columns in the dataset. If None, autofilled.
    pca     --> wether the columns need to be called Features or Components

    '''

    if columns is None and not pca:
        columns = ['Feature ' + str(i) for i in range(len(data[0]))]
    elif columns is None:
        columns = ['Component ' + str(i) for i in range(len(data[0]))]
    return pd.DataFrame(data, columns=columns)


def to_series(data, name=None):

    '''
    DESCRIPTION -----------------------------------

    Convert data to pd.Series.

    PARAMETERS -------------------------------------

    data --> dataset to convert
    name --> name of the target column. If None, autofilled.

    '''

    return pd.Series(data, name=name if name is not None else 'target')


def merge(X, y):
    ''' Merge pd.DataFrame and pd.Series into one df '''

    return X.merge(y.to_frame(), left_index=True, right_index=True)


def raise_TypeError(param, value):
    ''' Raise TypeError for wrong parameter type '''

    raise TypeError(f'Invalid type for {param} parameter: {type(value)}')


def raise_ValueError(param, value):
    ''' Raise ValueError for invalid parameter value '''

    raise ValueError(f'Invalid value for {param} parameter: {value}')


def check_isFit(isFit):
    if not isFit:
        raise AttributeError('You need to fit the class before calling ' +
                             'for a metric method!')


# << ============ Decorators ============ >>

def composed(*decs):
    ''' Add multiple decorators in one line '''

    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco


def timer(f):
    ''' Decorator to time a function '''

    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)

        # Get duration and print to log (args[0]=class instance)
        duration = time_to_string(start)
        prlog(f'Time elapsed: {duration}', args[0], 1)

        # Update class attribute
        if f.__name__ == 'fitting':
            args[0].fit_time = duration
        elif f.__name__ == 'bagging':
            args[0].bs_time = duration

        return result

    return wrapper


def params_to_log(f):
    ''' Decorator to save function's params to log file '''

    def wrapper(*args, **kwargs):
        log = args[0].T.log if hasattr(args[0], 'T') else args[0].log
        if log is not None:
            log.info('')  # Empty line first
            log.info(f'{args[0].__class__.__name__}.{f.__name__}. Parameters: {kwargs}')

        result = f(*args, **kwargs)
        return result

    return wrapper


def crash(f):
    ''' Decorator to save program crashes to log file '''

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

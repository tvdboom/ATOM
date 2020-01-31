# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Module containing utility functions.

'''


# << ============ Import Packages ============ >>

# Standard packages
import pandas as pd
import datetime
from time import time


# << ============ Functions ============ >>

def prlog(string, class_, level=0, time=False):

    '''
    DESCRIPTION -----------------------------------

    Print and save output to log file.

    PARAMETERS -------------------------------------

    string --> string to output
    class_ --> class of the element
    level  --> minimum verbosity level to print
    time   --> wether to add the timestamp to the log

    '''

    try:  # For the ATOM class
        verbose = class_.verbose
        log = class_.log
    except AttributeError:  # For the BaseModel class
        verbose = class_.T.verbose
        log = class_.T.log

    if verbose > level:
        print(string)

    if log is not None:
        with open(log, 'a+') as file:
            if time:
                # Datetime object containing current date and time
                now = datetime.now()
                date = now.strftime("%d/%m/%Y %H:%M:%S")
                file.write(date + '\n' + string + '\n')
            else:
                file.write(string + '\n')


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
        # args[0]=class instance
        prlog('Function "' + str(f.__name__) + f'" parameters: {kwargs}',
              args[0], 5)
        result = f(*args, **kwargs)
        return result

    return wrapper

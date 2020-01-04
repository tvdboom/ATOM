# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Module containing the metric class for all pre-defined metrics.


To add a new metric:
    1. Add the model's name to the __init__ method calling its function
    2. If necessary, create the adapted function above the class
    3. Add the mertic to mbin, mclass or mreg in atom.py
    3. Add the metric to the metrics dictionary in atom.py
    4. Add the metric's method to the ATOM class

'''

from sklearn.metrics import (
        accuracy_score, average_precision_score, roc_auc_score,
        mean_absolute_error, max_error, matthews_corrcoef, mean_squared_error,
        mean_squared_log_error, f1_score, hamming_loss, jaccard_score,
        log_loss, precision_score, r2_score, recall_score, confusion_matrix
        )


# << ============ Global variables ============ >>

# List of integer metrics
no_decimals = ['tn', 'fp', 'fn', 'tp']


# << ============ Functions ============ >>

def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).ravel()[0]


def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).ravel()[1]


def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).ravel()[2]


def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).ravel()[3]


def ap_adapted(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average='weighted')


def auc_adapted(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')


def f1_adapted(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def jaccard_adapted(y_true, y_pred):
    return jaccard_score(y_true, y_pred, average='weighted')


def precision_adapted(y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted')


def recall_adapted(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted')


# << ============ Classes ============ >>

class BaseMetric(object):

    def __init__(self, function, gib, needs_proba, task):

        '''
        DESCRIPTION -----------------------------------

        Class that contains all the information about the current metric.

        PARAMETERS -------------------------------------

        function    --> metric's function callable
        gib         --> if metric is a score function or a loss function
        needs_proba --> wether the metric needs a probability or score
        task        --> task of the main ATOM class

        '''

        # Get right function dependent on task
        if isinstance(function, str):
            if function == 'tn' and task == 'binary classification':
                self.function = tn
                self.longname = 'true negatives'
            elif function == 'fp' and task == 'binary classification':
                self.function = fp
                self.longname = 'false positives'
            elif function == 'fn' and task == 'binary classification':
                self.function = fn
                self.longname = 'false negatives'
            elif function == 'tp' and task == 'binary classification':
                self.function = tp
                self.longname = 'true positives'
            elif function == 'accuracy':
                self.function = accuracy_score
            elif function == 'ap':
                self.function = average_precision_score
            elif function == 'auc' and task == 'binary classification':
                self.function = roc_auc_score
            elif function == 'auc':
                self.function = auc_adapted
                self.longname = 'roc_auc_score'
            elif function == 'mae':
                self.function = mean_absolute_error
            elif function == 'max_error':
                self.function = max_error
            elif function == 'mcc':
                self.function = matthews_corrcoef
            elif function == 'mse':
                self.function = mean_squared_error
            elif function == 'msle':
                self.function = mean_squared_log_error
            elif function == 'f1' and task == 'binary classification':
                self.function = f1_score
            elif function == 'f1':
                self.function = f1_adapted
                self.longname = 'f1_score'
            elif function == 'hamming':
                self.function = hamming_loss
            elif function == 'jaccard' and task == 'binary classification':
                self.function = jaccard_score
            elif function == 'jaccard':
                self.function = jaccard_adapted
                self.longname = 'jaccard_score'
            elif function == 'logloss':
                self.function = log_loss
            elif function == 'precision' and task == 'binary classification':
                self.function = precision_score
            elif function == 'precision':
                self.function = precision_adapted
                self.longname = 'precision_score'
            elif function == 'r2':
                self.function = r2_score
            elif function == 'recall' and task == 'binary classification':
                self.function = recall_score
            elif function == 'recall':
                self.function = recall_adapted
                self.longname = 'recall_score'
            else:
                self.function = 'error'
                self.longname = function
        else:
            self.function = function  # Added by the user

        # Set rest of attributes
        if isinstance(function, str):
            self.name = function
        else:
            self.name = self.function.__name__
        if not hasattr(self, 'longname'):
            self.longname = self.function.__name__
        self.gib = gib
        self.needs_proba = needs_proba
        self.task = task
        self.dec = 0 if self.name in no_decimals else 4  # Number of decimals

    def func(self, y_true, y_pred):
        ''' Calculate the metric's score '''

        # For binary tasks returns the probability of class=1
        if self.task == 'binary classification' and self.needs_proba:
            return self.function(y_true, y_pred[:, 1])
        else:
            return self.function(y_true, y_pred)

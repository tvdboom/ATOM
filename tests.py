# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests.

'''

# Import packages
import numpy as np
import pandas as pd
import unittest
import multiprocessing
from sklearn.datasets import load_boston, load_wine, load_breast_cancer
from atom import ATOMClassifier, ATOMRegressor


# << ====================== Functions ====================== >>

def load_df(dataset):
    ''' Load dataset as pd.DataFrame '''

    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    data = pd.DataFrame(data, columns=columns)
    X = data.drop('target', axis=1)
    Y = data['target']
    return X, Y


def load_australian_dataset():
    ''' Load the Australian weather dataset for binary classification '''

    X = pd.read_csv('weatherAUS.csv', nrows=1e3)
    y = X['RainTomorrow']
    X = X.drop('RISK_MM', axis=1)  # Feature related to RainTomorrow
    X = X.drop('Date', axis=1)  # Irrelevant feature
    X = X.drop('RainTomorrow', axis=1)
    return X, y


# << ====================== Classes ====================== >>

class init(unittest.TestCase):

    # << ================== Test handling input data ================== >>

    def test_X_y_equal_length(self):
        ''' Test if error is raised when X and y don't have equal length '''

        X, _ = load_breast_cancer(return_X_y=True)
        y = [0, 0, 1, 1, 0]
        self.assertRaises(ValueError, ATOMClassifier, X, y)

    def test_y_is1dimensional(self):
        ''' Test if error is raised when y has only one dimension '''

        X, _ = load_breast_cancer(return_X_y=True)
        y = [[0, 0], [1, 1], [0, 1]]
        self.assertRaises(ValueError, ATOMClassifier, X, y)

    def test_isPandas(self):
        ''' Test if certain data attributes are pd.DataFrames or pd.Series '''

        def test(atom):
            for attr in ['dataset', 'train', 'test', 'X', 'X_train', 'X_test']:
                self.assertIsInstance(getattr(atom, attr), pd.DataFrame)

            for attr in ['y', 'y_train', 'y_test']:
                self.assertIsInstance(getattr(atom, attr), pd.Series)

        # Test with lists
        X = [[0, 1], [2, 2], [0, 1]]
        y = [0, 1, 0]
        atom = ATOMClassifier(X, y)
        test(atom)

        # Test with np.arrays
        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        test(atom)

    def test_merger_X_y(self):
        ''' Test that merger between X and y was successfull '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertEqual(len(atom.X.columns) + 1, len(atom.dataset.columns))

    def test_target_column_last(self):
        ''' Test if target column is placed last '''

        # When it is last, if self.target is assigned correctly
        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertEqual(atom.dataset.columns[-1], atom.target)

        # When it's not last, if it is moved to the last position correctly
        X, y = load_australian_dataset()
        atom = ATOMClassifier(X, target='MaxTemp')
        self.assertEqual(atom.dataset.columns[-1], atom.target)

    def test_target_in_data(self):
        ''' Test if the target column is in the X dataframe '''

        X, y = load_australian_dataset()
        self.assertRaises(ValueError, ATOMClassifier, X, target='not_there')

    # << ====================== Test parameters ====================== >>

    def test_isfit_attribute(self):
        ''' Test if the _isfit attribute is set correctly '''

        X, y = load_australian_dataset()
        atom = ATOMClassifier(X, y)
        self.assertFalse(atom._isfit)
        atom.impute()
        atom.encode()
        atom.fit('LR', 'f1', max_iter=0, bagging=0)
        self.assertTrue(atom._isfit)

    def test_percentage_parameter(self):
        ''' Test if the percentage parameter is set correctly '''

        # Test if it changes for forbidden values
        X, y = load_breast_cancer(return_X_y=True)
        for percentage in [0, -1, 120]:
            atom = ATOMClassifier(X, y, percentage=percentage)
            self.assertEqual(atom.percentage, 100)

        # Test if it works correctly
        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y, percentage=10)
        self.assertEqual(len(atom.X), int(len(X) * 0.10))
        atom = ATOMClassifier(X, y, percentage=48)
        self.assertEqual(len(atom.y), int(len(y) * 0.48))

    def test_njobs_parameter(self):
        ''' Test n_jobs to be correct '''

        X, y = load_breast_cancer(return_X_y=True)

        n_cores = multiprocessing.cpu_count()
        for n_jobs in [12, -1, -2, 0]:
            atom = ATOMClassifier(X, y, n_jobs=n_jobs)
            self.assertTrue(0 < atom.n_jobs <= n_cores)

        # Assert error when input is invalid
        self.assertRaises(ValueError, ATOMClassifier, X, y, n_jobs=-200)


class reset_attributes(unittest.TestCase):

    def test_attributes_equal_length(self):
        ''' Test if certain data attributes have the same number of rows '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)

        attr1, attr2 = ['X', 'X_train', 'X_test'], ['y', 'y_train', 'y_test']
        for df1, df2 in zip(attr1, attr2):
            self.assertEqual(len(getattr(atom, df1)), len(getattr(atom, df2)))


if __name__ == '__main__':

    unittest.main()

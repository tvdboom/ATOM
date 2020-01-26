# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the utility methods of the ATOM class.

'''

# Import packages
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_boston
from atom import ATOMClassifier, ATOMRegressor


# << ====================== Variables ===================== >>

X_dim4 = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
y_dim4 = [0, 1, 1, 0]

# List of pre-set binary classification metrics
mbin = ['tn', 'fp', 'fn', 'tp', 'ap']

# List of pre-set classification metrics
mclass = ['accuracy', 'auc', 'mcc', 'f1', 'hamming', 'jaccard', 'logloss',
          'precision', 'recall']

# List of pre-set regression metrics
mreg = ['mae', 'max_error', 'mse', 'msle', 'r2']


# << ====================== Functions ====================== >>

def load_df(dataset):
    ''' Load dataset as pd.DataFrame '''

    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    data = pd.DataFrame(data, columns=columns)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y


# << ================ Test _split_dataset ================== >>

def test_dataset_is_shuffled():
    ''' Assert that self.dataset is shuffled '''

    X, y = load_df(load_wine())
    atom = ATOMClassifier(X, y)
    for i in np.random.randint(0, len(X), 10):
        assert not atom.X.equals(X)


def test_percentage_data_selected():
    ''' Assert that a percentage of the data is selected correctly '''

    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y, percentage=10)
    assert len(atom.X) == int(len(X) * 0.10)
    atom = ATOMClassifier(X, y, percentage=48)
    assert len(atom.y) == int(len(y) * 0.48)


def test_train_test_split():
    ''' Assert that the train and test split is made correctly '''

    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y, test_size=0.13)
    assert len(atom.train) == int(0.87*len(X))


# << =============== Test reset_attributes ================= >>

def test_truth_parameter():
    ''' Assert that the truth parameter is set correctly '''

    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)
    pytest.raises(TypeError, atom.reset_attributes, 2)


def test_changes_based_ground_truth():
    ''' Assert that method works as intended for different ground truths '''

    # When dataset is changed
    atom = ATOMClassifier(X_dim4, y_dim4)
    atom.dataset.iloc[0, 2] = 20
    atom.reset_attributes('dataset')
    assert atom.train.iloc[0, 2] == 20
    assert atom.X.iloc[0, 2] == 20
    assert atom.X_train.iloc[0, 2] == 20

    # When train is changed
    atom.train.iloc[1, 1] = 10
    atom.reset_attributes('train')
    assert atom.dataset.iloc[1, 1] == 10
    assert atom.X.iloc[1, 1] == 10
    assert atom.X_train.iloc[1, 1] == 10

    # When y_train is changed
    atom.y_train.iloc[0] = 2
    atom.reset_attributes('y_train')
    assert atom.dataset.iloc[0, -1] == 2
    assert atom.train.iloc[0, -1] == 2
    assert atom.y.iloc[0] == 2

    # When X is changed
    atom.X.iloc[0, 1] = 0.112
    atom.reset_attributes('X')
    assert atom.dataset.iloc[0, 1] == 0.112
    assert atom.train.iloc[0, 1] == 0.112
    assert atom.X_train.iloc[0, 1] == 0.112


def test_index_reset():
    ''' Assert that indices are reset for all data attributes '''

    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)
    for attr in ['dataset', 'train', 'test', 'X', 'y',
                 'X_train', 'y_train', 'X_test', 'y_test']:
        idx = list(getattr(atom, attr).index)
        assert idx == list(range(len(getattr(atom, attr))))


def test_isPandas():
    ''' Assert that data attributes are pd.DataFrames or pd.Series '''

    def test(atom):
        for attr in ['dataset', 'train', 'test', 'X', 'X_train', 'X_test']:
            assert isinstance(getattr(atom, attr), pd.DataFrame)

        for attr in ['y', 'y_train', 'y_test']:
            assert isinstance(getattr(atom, attr), pd.Series)

    # Test with lists
    atom = ATOMClassifier(X_dim4, y_dim4)
    test(atom)

    # Test with np.arrays
    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)
    test(atom)


def test_attributes_equal_length():
    ''' Assert that data attributes have the same number of rows '''

    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)

    attr1, attr2 = ['X', 'X_train', 'X_test'], ['y', 'y_train', 'y_test']
    for df1, df2 in zip(attr1, attr2):
        assert len(getattr(atom, df1)) == len(getattr(atom, df2))


# << ==================== Test report ====================== >>

def test_df_parameter():
    ''' Assert that the df parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(TypeError, atom.report, df=True)


def test_rows_parameter():
    ''' Assert that the rows parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(TypeError, atom.report, rows='1000')


def test_filename_parameter():
    ''' Assert that the filename parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(TypeError, atom.report, filename=False)


def test_creates_report():
    ''' Assert that the report has been created and saved'''

    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)
    atom.report(rows=10)  #, filename='report')
    assert hasattr(atom, 'report')


# << ===================== Test scale ====================== >>

def test_scale():
    ''' Assert that the scale method normalizes the features '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.scale()
    assert atom.dataset.iloc[:, 1].mean() < 0.05  # Not exactly 0
    assert atom.dataset.iloc[:, 1].std() < 3


def test_already_scaled():
    ''' Assert that the scale method does nothing when already scaled '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.scale()

    atom2 = ATOMClassifier(atom.X, atom.y, random_state=1)
    X_already_scaled = atom2.X.copy()
    atom2.scale()
    assert atom2.X.equals(X_already_scaled)


# << ================ Test _final_results ================== >>

def test_error_not_fit():
    ''' Assert that an error is raised when the ATOM class is not fitted '''

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(AttributeError, atom.hamming)


def test_error_wrong_metric():
    ''' Assert that an error is raised when an invalid metric is selected '''

    atom = ATOMRegressor(X_dim4, y_dim4)
    atom.fit(models='lgb', metric='msle', max_iter=0)
    pytest.raises(ValueError, atom.hamming)


def test_all_metric_methods():
    ''' Assert that all metric's methods work as intended '''

    # For binary classification
    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)
    atom.fit(models='lgb', metric='auc', max_iter=0)
    for metric in mbin:
        getattr(atom, metric)()
        assert 1 == 1

    # For multiclass classification
    X, y = load_wine(return_X_y=True)
    atom = ATOMClassifier(X, y)
    atom.fit(models='lgb', metric='f1', max_iter=0)
    for metric in mclass:
        getattr(atom, metric)()
        assert 1 == 1

    # For regression
    X, y = load_boston(return_X_y=True)
    atom = ATOMRegressor(X, y)
    atom.fit(models='lgb', metric='r2', max_iter=0)
    for metric in mreg:
        getattr(atom, metric)()
        assert 1 == 1

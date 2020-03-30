# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the impute method of the ATOM class.

'''

# Import packages
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from atom import ATOMClassifier


# << ====================== Functions ====================== >>

def load_df(dataset):
    ''' Load dataset as pd.DataFrame '''

    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    data = pd.DataFrame(data, columns=columns)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y


# << ====================== Variables ===================== >>

X_dim4 = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
X_dim4_missing = [[np.nan, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
X_str = [[np.nan, '1', '1'], ['2', '5', '3'], ['2', '1', '3'], ['3', '1', '2']]
y_dim4 = [0, 1, 0, 1]


# << ======================= Tests ========================= >>

# << =================== Test parameters =================== >>

def test_strat_num_parameter():
    ''' Assert that the strat_num parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(ValueError, atom.impute, strat_num='test')


def test_min_frac_rows_parameter():
    ''' Assert that the min_frac_rows parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(ValueError, atom.impute, min_frac_rows=1.0)


def test_min_frac_cols_parameter():
    ''' Assert that the min_frac_cols parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(ValueError, atom.impute, min_frac_cols=5.2)


def test_missing_string():
    ''' Assert that the missing parameter handles only a string correctly '''

    X = [[4, 1, 2], [3, 1, 2], ['r', 'a', 'b'], [2, 1, 1]]
    atom = ATOMClassifier(X, y_dim4)
    atom.impute(strat_num='remove', strat_cat='remove', missing='a')
    assert atom.dataset.isna().sum().sum() == 0


def test_missing_extra_values():
    ''' Assert that the missing parameter adds obligatory values '''

    X = [[np.nan, 1, 2], [None, 1, 2], ['', 'a', 'b'], [2, np.inf, 1]]
    atom = ATOMClassifier(X, y_dim4)
    atom.impute(strat_num='remove', strat_cat='remove', missing=['O', 'N'])
    assert atom.dataset.isna().sum().sum() == 0


def test_imputing_all_missing_values():
    ''' Assert that all missing values are imputed in str and numeric '''

    missing = [np.inf, -np.inf, '', '?', 'NA', 'nan', 'inf']
    for v in missing:
        # Create new inputs with different missing values
        Xs = [[[v, 1, 1], [2, 5, 2], [4, v, 1], [2, 1, 1]],
              [[v, '1', '1'], ['2', '5', v], ['2', '1', '3'], ['3', '1', '1']]]
        for X in Xs:
            atom = ATOMClassifier(X, y_dim4, random_state=1)
            atom.impute(strat_num='mean')
            assert atom.dataset.isna().sum().sum() == 0


# << ========= Test too many NaNs in rows and cols ========= >>

def test_rows_too_many_nans():
    ''' Assert that rows with too many NaN values are dropped '''

    X, y = load_df(load_breast_cancer())
    for i in range(5):  # Add 5 rows with all NaN values
        X.loc[len(X)] = [np.nan for _ in range(X.shape[1])]
        y.loc[len(X)] = 1
    atom = ATOMClassifier(X, y)
    atom.impute(strat_num='mean', strat_cat='most_frequent')
    assert len(atom.dataset) == 569  # 569 is original length
    assert atom.dataset.isna().sum().sum() == 0


def test_cols_too_many_nans():
    ''' Assert that columns with too many NaN values are dropped '''

    X, y = load_df(load_breast_cancer())
    for i in range(5):  # Add 5 cols with all NaN values
        X['col ' + str(i)] = [np.nan for _ in range(X.shape[0])]
    atom = ATOMClassifier(X, y)
    atom.impute(strat_num='mean', strat_cat='most_frequent')
    assert len(atom.X.columns) == 30  # Original number of cols
    assert atom.dataset.isna().sum().sum() == 0


# << ================ Test numeric columns ================ >>

def test_imputing_numeric_remove():
    ''' Assert that imputing remove for numerical values works '''

    atom = ATOMClassifier(X_dim4_missing, y_dim4, random_state=1)
    atom.impute(strat_num='remove')
    assert len(atom.dataset) == 3
    assert atom.dataset.isna().sum().sum() == 0


def test_imputing_numeric_number():
    ''' Assert that imputing a number for numerical values works '''

    atom = ATOMClassifier(X_dim4_missing, y_dim4, random_state=1)
    atom.impute(strat_num=3.2)
    assert atom.dataset.iloc[2, 0] == 3.2
    assert atom.dataset.isna().sum().sum() == 0


def test_imputing_numeric_knn():
    ''' Assert that imputing numerical values with KNNImputer works '''

    atom = ATOMClassifier(X_dim4_missing, y_dim4, random_state=1)
    atom.impute(strat_num='knn')
    assert atom.dataset.iloc[3, 0] == 2
    assert atom.dataset.isna().sum().sum() == 0


def test_imputing_numeric_mean():
    ''' Assert that imputing the mean for numerical values works '''

    atom = ATOMClassifier(X_dim4_missing, y_dim4, random_state=1)
    atom.impute(strat_num='mean')
    assert atom.dataset.iloc[2, 0] == 6.5
    assert atom.dataset.isna().sum().sum() == 0


def test_imputing_numeric_median():
    ''' Assert that imputing the median for numerical values works '''

    atom = ATOMClassifier(X_dim4_missing, y_dim4, random_state=1)
    atom.impute(strat_num='median')
    assert atom.dataset.iloc[2, 0] == 6.5
    assert atom.dataset.isna().sum().sum() == 0


def test_imputing_numeric_most_frequent():
    ''' Assert that imputing the most_frequent for numerical values works '''

    atom = ATOMClassifier(X_dim4_missing, y_dim4, random_state=1)
    atom.impute(strat_num='most_frequent')
    assert atom.dataset.iloc[2, 0] == 5
    assert atom.dataset.isna().sum().sum() == 0


# << ================ Test non-numeric columns ================ >>

def test_imputing_non_numeric_string():
    ''' Test imputing a string for non-numerical '''

    atom = ATOMClassifier(X_str, y_dim4, random_state=1)
    atom.impute(strat_num='mean', strat_cat='3.2')
    assert atom.dataset.iloc[2, 0] == '3.2'
    assert atom.dataset.isna().sum().sum() == 0


def test_imputing_non_numeric_remove():
    ''' Test imputing remove for non-numerical '''

    atom = ATOMClassifier(X_str, y_dim4, random_state=1)
    atom.impute(strat_num='knn', strat_cat='remove')
    assert len(atom.dataset) == 3
    assert atom.dataset.isna().sum().sum() == 0


def test_imputing_non_numeric_most_frequent():
    ''' Test imputing most_frequent for non-numerical '''

    atom = ATOMClassifier(X_str, y_dim4, random_state=1)
    atom.impute(strat_num='knn', strat_cat='most_frequent')
    assert atom.dataset.iloc[2, 0] == '2'
    assert atom.dataset.isna().sum().sum() == 0

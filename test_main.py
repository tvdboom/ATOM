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
    y = data['target']
    return X, y


# << ====================== Classes ====================== >>

class init(unittest.TestCase):

    # << ================== Test handling input data ================== >>

    def test_X_type(self):
        ''' Test if error when X is wrong type '''

        self.assertRaises(TypeError, ATOMClassifier, X=23.2)

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

    def test_merger_X_y(self):
        ''' Test that merger between X and y was successfull '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertEqual(len(atom.X.columns) + 1, len(atom.dataset.columns))
        self.assertEqual(len(X), len(atom.dataset))

    def test_target_attribute_and_placement(self):
        ''' Test if target column is assigned correctly and placed last '''

        # When it is last, if self.target is assigned correctly
        X, y = load_df(load_breast_cancer())
        atom = ATOMClassifier(X, y)
        self.assertEqual(atom.target, y.name)
        self.assertEqual(atom.dataset.columns[-1], atom.target)

        # When y is None...
        atom = ATOMClassifier(X)
        self.assertEqual(atom.dataset.columns[-1], atom.target)

        # When it's not last, if it is moved to the last position correctly
        X, y = load_df(load_breast_cancer())
        atom = ATOMClassifier(X, y='mean texture')
        self.assertEqual(atom.target, 'mean texture')
        self.assertEqual(atom.dataset.columns[-1], atom.target)

    def test_target_in_data(self):
        ''' Test if the target column given by y is in X '''

        X, y = load_df(load_breast_cancer())
        self.assertRaises(ValueError, ATOMClassifier, X, y='test')

    def test_y_type(self):
        ''' Test if error when y is wrong type '''

        X, y = load_breast_cancer(return_X_y=True)
        self.assertRaises(TypeError, ATOMClassifier, X, y=23.2)

    # << ====================== Test parameters ====================== >>

    def test_isfit_attribute(self):
        ''' Test if the _isfit attribute is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertFalse(atom._isfit)
        atom.impute()
        atom.encode()
        atom.fit('LR', 'f1', max_iter=0, bagging=0)
        self.assertTrue(atom._isfit)

    def test_percentage_parameter(self):
        ''' Test if the percentage parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        self.assertRaises(TypeError, ATOMClassifier, X, y, percentage='test')
        for i in [0, -1, 120]:
            self.assertRaises(ValueError, ATOMClassifier, X, y, percentage=i)

    def test_test_size_parameter(self):
        ''' Test if the test_size parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        self.assertRaises(TypeError, ATOMClassifier, X, y, test_size=3)
        for i in [0., -3.1, 12.2]:
            self.assertRaises(ValueError, ATOMClassifier, X, y, test_size=i)

    def test_log_parameter(self):
        ''' Test if the log parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        self.assertRaises(TypeError, ATOMClassifier, X, y, log=3)

    def test_warnings_parameter(self):
        ''' Test if the warnings parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        self.assertRaises(TypeError, ATOMClassifier, X, y, warnings='True')

        # Check if set True when 1
        atom = ATOMClassifier(X, y, warnings=1)
        self.assertTrue(atom.warnings)

    def test_verbose_parameter(self):
        ''' Test if the verbose parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        self.assertRaises(TypeError, ATOMClassifier, X, y, verbose=3.2)
        for i in [-2, 4]:
            self.assertRaises(ValueError, ATOMClassifier, X, y, verbose=i)

    def test_random_state_parameter(self):
        ''' Test if the random_state parameter is set correctly and works '''

        X, y = load_breast_cancer(return_X_y=True)
        self.assertRaises(TypeError, ATOMClassifier, X, y, verbose=3.2)

        # Check if it gives the same results every time
        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y, n_jobs=-1, random_state=1)
        atom.fit(models=['lr', 'lgb', 'pa'],
                 metric='f1',
                 max_iter=3,
                 cv=1,
                 bagging=0)
        atom2 = ATOMClassifier(X, y, n_jobs=-1, random_state=1)
        atom2.fit(models=['lr', 'lgb', 'pa'],
                  metric='f1',
                  max_iter=3,
                  cv=1,
                  bagging=0)
        self.assertEqual(atom.lr.score_test, atom2.lr.score_test)
        self.assertEqual(atom.lgb.score_test, atom2.lgb.score_test)
        self.assertEqual(atom.pa.score_test, atom2.pa.score_test)

    def test_njobs_parameter(self):
        ''' Test if the n_jobs parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)

        # Test if error raises for forbidden types or values
        self.assertRaises(TypeError, ATOMClassifier, X, y, n_jobs='test')
        self.assertRaises(ValueError, ATOMClassifier, X, y, n_jobs=-200)

        n_cores = multiprocessing.cpu_count()
        for n_jobs in [59, -1, -2, 0]:
            atom = ATOMClassifier(X, y, n_jobs=n_jobs)
            self.assertTrue(0 < atom.n_jobs <= n_cores)

    # << ==================== Test data cleaning ==================== >>

    def test_remove_invalid_column_type(self):
        ''' Test if self.dataset removes invalid columns types '''

        X, y = load_df(load_breast_cancer())
        # Make datetime column
        X['invalid_column'] = pd.to_datetime(X['mean radius'])
        atom = ATOMClassifier(X, y)
        self.assertFalse('invalid_column' in atom.dataset.columns)

    def test_remove_maximum_cardinality(self):
        ''' Test if self.dataset removes columns with maximum cardinality '''

        X, y = load_df(load_breast_cancer())
        # Create column with all different values
        X['invalid_column'] = [str(i) for i in range(len(X))]
        atom = ATOMClassifier(X, y)
        self.assertFalse('invalid_column' in atom.dataset.columns)

    def test_raise_one_target_value(self):
        ''' Test if error raises when there is only 1 target value '''

        X, y = load_breast_cancer(return_X_y=True)
        y = [1 for _ in range(len(y))]  # All targets are equal to 1
        self.assertRaises(ValueError, ATOMClassifier, X, y)

    def test_remove_minimum_cardinality(self):
        ''' Test if self.dataset removes columns with only 1 value '''

        X, y = load_df(load_breast_cancer())
        # Create column with all different values
        X['invalid_column'] = [2.3 for i in range(len(X))]
        atom = ATOMClassifier(X, y)
        self.assertFalse('invalid_column' in atom.dataset.columns)

    def test_remove_duplicate_rows(self):
        ''' Test if self.dataset removes duplicate rows '''

        X, y = load_df(load_breast_cancer())
        len_ = len(X)  # Save number of non-duplicate rows
        for i in range(5):  # Add 5 rows with exactly the same values
            X.loc[len(X)] = X.iloc[i, :]
            y.loc[len(X)] = y.iloc[i]
        atom = ATOMClassifier(X, y)
        self.assertEqual(len_, len(atom.dataset))

    def test_remove_rows_nan_target(self):
        ''' Test if self.dataset removes rows with NaN in target column '''

        X, y = load_df(load_breast_cancer())
        len_ = len(X)  # Save number of non-duplicate rows
        y[0], y[21] = np.NaN, np.NaN  # Set NaN to target column for 2 rows
        atom = ATOMClassifier(X, y)
        self.assertEqual(len_, len(atom.dataset) + 2)

    # << ==================== Test task assigning ==================== >>

    def test_task_assigning(self):
        ''' Test if self.task is assigned correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertEqual(atom.task, 'binary classification')

        X, y = load_wine(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertEqual(atom.task, 'multiclass classification')

        X, y = load_boston(return_X_y=True)
        atom = ATOMRegressor(X, y)
        self.assertEqual(atom.task, 'regression')

    # << ================ Test mapping target column ================ >>

    def test_encode_target_column(self):
        ''' Test the encoding of the target column '''

        X = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
        y = ['y', 'n', 'y', 'n']
        atom = ATOMClassifier(X, y)
        self.assertTrue(atom.dataset[atom.target].dtype.kind == 'i')

    def test_target_mapping(self):
        ''' Test if target_mapping attribute is set correctly '''

        X = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
        y = ['y', 'n', 'y', 'n']
        atom = ATOMClassifier(X, y)
        self.assertEqual(atom.mapping, dict(n=0, y=1))

        X, y = load_wine(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertEqual(atom.mapping, {'0': 0, '1': 1, '2': 2})

        X, y = load_boston(return_X_y=True)
        atom = ATOMRegressor(X, y)
        self.assertTrue(isinstance(atom.mapping, str))


class _split_dataset(unittest.TestCase):

    def test_dataset_is_shuffled(self):
        ''' Test if self.dataset is shuffled '''

        X, y = load_df(load_wine())
        atom = ATOMClassifier(X, y)
        for i in np.random.randint(0, len(X), 10):
            self.assertNotEqual(atom.dataset.iloc[i, :].values,
                                X.iloc[i, :].values)

    def test_percentage_data_selected(self):
        ''' Test if a percentage of the data is selected correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y, percentage=10)
        self.assertEqual(len(atom.X), int(len(X) * 0.10))
        atom = ATOMClassifier(X, y, percentage=48)
        self.assertEqual(len(atom.y), int(len(y) * 0.48))

    def test_train_test_split(self):
        ''' Test if the train and test split is made correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y, test_size=0.13)
        self.assertEqual(len(atom.train), int(0.87*len(X)))


class reset_attributes(unittest.TestCase):

    def test_truth_parameter(self):
        ''' Test if the truth parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.reset_attributes, 2)

    def test_changes_based_ground_truth(self):
        ''' Test if method works as intended for different ground truths '''

        X = [[0, 1], [1, 1], [1, 2]]
        y = [0, 0, 1]

        # When dataset is changed
        atom = ATOMClassifier(X, y)
        atom.dataset['Feature 0'][0] = 20
        atom.reset_attributes()
        self.assertEqual(atom.train['Feature 0'][0], 20)
        self.assertEqual(atom.X['Feature 0'][0], 20)
        self.assertEqual(atom.X_train['Feature 0'][0], 20)

        # When train is changed
        atom = ATOMClassifier(X, y)
        atom.train['Feature 1'][1] = 10
        atom.reset_attributes('train')
        self.assertEqual(atom.dataset['Feature 1'][1], 10)
        self.assertEqual(atom.X['Feature 1'][1], 10)
        self.assertEqual(atom.X_train['Feature 1'][1], 10)

        # When y_train is changed
        atom = ATOMClassifier(X, y)
        atom.y_train[0] = 2
        atom.reset_attributes('y_train')
        self.assertEqual(atom.dataset.target[0], 2)
        self.assertEqual(atom.train.target[0], 2)
        self.assertEqual(atom.y[0], 2)

        # When X is changed
        atom = ATOMClassifier(X, y)
        atom.X['Feature 0'] = 0.112
        atom.reset_attributes('X')
        self.assertEqual(atom.dataset['Feature 0'][0], 0.112)
        self.assertEqual(atom.train['Feature 0'][0], 0.112)
        self.assertEqual(atom.X_train['Feature 0'][0], 0.112)

    def test_index_reset(self):
        ''' Test if indices are reset for all data attributes '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        for attr in ['dataset', 'train', 'test', 'X', 'y',
                     'X_train', 'y_train', 'X_test', 'y_test']:
            self.assertEqual(list(getattr(atom, attr).index),
                             list(range(len(getattr(atom, attr)))))

    def test_isPandas(self):
        ''' Test if data attributes are pd.DataFrames or pd.Series '''

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

    def test_attributes_equal_length(self):
        ''' Test if data attributes have the same number of rows '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)

        attr1, attr2 = ['X', 'X_train', 'X_test'], ['y', 'y_train', 'y_test']
        for df1, df2 in zip(attr1, attr2):
            self.assertEqual(len(getattr(atom, df1)), len(getattr(atom, df2)))


class report(unittest.TestCase):

    def test_df_parameter(self):
        ''' Test if the df parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.report, df=True)

    def test_rows_parameter(self):
        ''' Test if the rows parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.report, rows='1000')

    def test_filename_parameter(self):
        ''' Test if the filename parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.report, filename=False)

    def test_creates_report(self):
        ''' Test if the report has been created '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        atom.report(rows=10)
        self.assertTrue(hasattr(atom, 'report'))


class impute(unittest.TestCase):

    # << ================ Test parameters ================ >>

    def test_strat_num_parameter(self):
        ''' Test if the strat_num parameter is set correctly '''

        X = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.impute, strat_num=[2, 2])
        self.assertRaises(ValueError, atom.impute, strat_num='test')

    def test_strat_cat_parameter(self):
        ''' Test if the strat_cat parameter is set correctly '''

        X = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.impute, strat_cat=1.2)

    def test_max_frac_rows_parameter(self):
        ''' Test if the mac_frac_rows parameter is set correctly '''

        X = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.impute, max_frac_rows=2)
        self.assertRaises(ValueError, atom.impute, max_frac_rows=1.0)

    def test_max_frac_cols_parameter(self):
        ''' Test if the mac_frac_cols parameter is set correctly '''

        X = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.impute, max_frac_cols='test')
        self.assertRaises(ValueError, atom.impute, max_frac_cols=5.2)

    def test_missing_string(self):
        ''' Test if the missing parameter handles only a string correctly '''

        X = [[4, 1, 2], [3, 1, 2], ['r', 'a', 'b'], [2, 1, 1]]
        y = [1, 1, 0, 1]
        atom = ATOMClassifier(X, y)
        atom.impute(strat_num='remove', strat_cat='remove', missing='a')
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    def test_missing_extra_values(self):
        ''' Test if the missing parameter adds obligatory values '''

        X = [[np.nan, 1, 2], [None, 1, 2], ['', 'a', 'b'], [2, np.inf, 1]]
        y = [1, 1, 0, 1]
        atom = ATOMClassifier(X, y)
        atom.impute(strat_num='remove', strat_cat='remove', missing=['O', 'N'])
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    def test_imputing_all_missing_values(self):
        ''' Test if all missing values are imputed in str and numeric '''

        missing = [np.inf, -np.inf, '', '?', 'NA', 'nan', 'inf']
        for value in missing:
            # Create new inputs with different missing values
            Xs = [[[value, 1, 1], [2, 5, 2], [4, value, 1]],
                  [[value, '1', '1'], ['2', '5', value], ['2', '1', '3']]]
            for X in Xs:
                y = [0, 1, 0]
                atom = ATOMClassifier(X, y, random_state=1)
                atom.impute(strat_num='mean')
                self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    # << ========= Test too many NaNs in rows and cols ========= >>

    def test_rows_too_many_nans(self):
        ''' Test if rows with too many NaN values are dropped '''

        X, y = load_df(load_breast_cancer())
        for i in range(5):  # Add 5 rows with all NaN values
            X.loc[len(X)] = [np.nan for _ in range(X.shape[1])]
            y.loc[len(X)] = 1
        atom = ATOMClassifier(X, y)
        atom.impute(strat_num='mean', strat_cat='most_frequent')
        self.assertEqual(len(atom.dataset), 569)  # 569 is original length
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    def test_cols_too_many_nans(self):
        ''' Test if columns with too many NaN values are dropped '''

        X, y = load_df(load_breast_cancer())
        for i in range(5):  # Add 5 cols with all NaN values
            X['col ' + str(i)] = [np.nan for _ in range(X.shape[0])]
        atom = ATOMClassifier(X, y)
        atom.impute(strat_num='mean', strat_cat='most_frequent')
        self.assertEqual(len(atom.X.columns), 30)  # Original number of cols
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    # << ================ Test numeric columns ================ >>

    def test_imputing_numeric_remove(self):
        ''' Test if imputing remove for numerical values works '''

        X = [[np.nan, 1, 1], [2, 5, 2], [4, 2, 1], [4, 5, 1]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y, random_state=1)
        atom.impute(strat_num='remove')
        self.assertEqual(len(atom.dataset), 3)
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    def test_imputing_numeric_number(self):
        ''' Test if imputing a number for numerical values works '''

        X = [[np.nan, 0, 1], [2, 3, 4], [5, np.nan, 7], [8, 9, 10]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y, random_state=1)
        atom.impute(strat_num=3.2)
        self.assertEqual(atom.dataset.iloc[1, 1], 3.2)
        self.assertEqual(atom.dataset.iloc[2, 0], 3.2)
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    def test_imputing_numeric_knn(self):
        ''' Test if imputing numerical values with KNNImputer works '''

        X = [[np.nan, 0, 1], [2, 3, 4], [5, np.nan, 7], [8, 9, 10], [0, 3, 1]]
        y = [0, 1, 0, 1, 1]
        atom = ATOMClassifier(X, y, random_state=1)
        atom.impute(strat_num='knn')
        self.assertEqual(atom.dataset.iloc[0, 1], 3)
        self.assertAlmostEqual(atom.dataset.iloc[3, 0], 2.3333333333333335)
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    def test_imputing_numeric_mean(self):
        ''' Test if imputing the mean for numerical values works '''

        X = [[np.nan, 0, 1], [2, 3, 4], [5, np.nan, 7], [8, 9, 10]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y, random_state=1)
        atom.impute(strat_num='mean')
        self.assertEqual(atom.dataset.iloc[1, 1], 9)
        self.assertEqual(atom.dataset.iloc[2, 0], 6.5)
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    def test_imputing_numeric_median(self):
        ''' Test if imputing the median for numerical values works '''

        X = [[np.nan, 0, 1], [2, 3, 4], [5, np.nan, 7], [8, 9, 10]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y, random_state=1)
        atom.impute(strat_num='median')
        self.assertEqual(atom.dataset.iloc[1, 1], 9)
        self.assertEqual(atom.dataset.iloc[2, 0], 6.5)
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    def test_imputing_numeric_most_frequent(self):
        ''' Test if imputing the most_frequent for numerical values works '''

        X = [[np.nan, 0, 1], [2, 3, 4], [5, np.nan, 7], [8, 9, 10]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y, random_state=1)
        atom.impute(strat_num='most_frequent')
        self.assertEqual(atom.dataset.iloc[1, 1], 9)
        self.assertEqual(atom.dataset.iloc[2, 0], 5)
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    # << ================ Test non-numeric columns ================ >>

    def test_imputing_non_numeric_string(self):
        ''' Test imputing a string for non-numerical '''

        X = [[np.nan, '1', '1'], ['2', '5', '3'],
             ['2', '1', '3'], ['3', '1', '3']]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y, random_state=1)
        atom.impute(strat_num='mean', strat_cat='3.2')
        self.assertEqual(atom.dataset.iloc[2, 0], '3.2')
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    def test_imputing_non_numeric_remove(self):
        ''' Test imputing remove for non-numerical '''

        X = [[np.nan, '1', '1'], ['2', '5', '3'],
             ['2', '1', '3'], ['3', '1', '3']]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y, random_state=1)
        atom.impute(strat_num='knn', strat_cat='remove')
        self.assertEqual(len(atom.dataset), 3)
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)

    def test_imputing_non_numeric_most_frequent(self):
        ''' Test imputing most_frequent for non-numerical '''

        X = [[np.nan, '1', '1'], ['2', '5', '3'],
             ['2', '1', '3'], ['3', '1', '3']]
        y = [0, 1, 0, 0]
        atom = ATOMClassifier(X, y, random_state=1)
        atom.impute(strat_num='knn', strat_cat='most_frequent')
        self.assertEqual(atom.dataset.iloc[2, 0], '2')
        self.assertEqual(atom.dataset.isna().sum().sum(), 0)


class encode(unittest.TestCase):

    # << ================ Test parameters ================ >>

    def test_max_onehot_parameter(self):
        ''' Test if the max_onehot parameter is set correctly '''

        X = [[2, 0, 1], [2, 1, 2], [1, 0, 1], [2, 1, 0]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.encode, max_onehot=3.0)
        self.assertRaises(ValueError, atom.encode, max_onehot=-2)

    def test_frac_to_other_parameter(self):
        ''' Test if the frac_to_other parameter is set correctly '''

        X = [[2, 0, 1], [2, 1, 2], [1, 0, 1], [2, 1, 0]]
        y = [0, 1, 0, 1]
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.encode, frac_to_other=2)
        self.assertRaises(ValueError, atom.encode, frac_to_other=2.2)

    def test_frac_to_other(self):
        ''' Test if the other values are created when encoding '''

        X = [[2, 0, 'a'], [2, 3, 'a'], [5, 2, 'b'], [1, 2, 'a'], [1, 2, 'c'],
             [2, 0, 'd'], [2, 3, 'd'], [5, 2, 'd'], [1, 2, 'a'], [1, 2, 'd']]
        y = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
        atom = ATOMClassifier(X, y)
        atom.encode(max_onehot=5, frac_to_other=0.3)
        self.assertTrue('Feature 2_other' in atom.dataset.columns)

    # << ================ Test encoding types ================ >>

    def test_label_encoder(self):
        ''' Test if the label-encoder works as intended '''

        X = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
        y = ['a', 'b', 'a', 'b']
        atom = ATOMClassifier(X, y)

        # Get the values of the target feature and check it's only 0 and 1
        keys = atom.dataset['target'].value_counts().keys()[0]
        self.assertTrue((keys == [0, 1]).sum(), 2)

    def test_one_hot_encoder(self):
        ''' Test if the one-hot-encoder works as intended '''

        X = [[2, 0, 'a'], [2, 3, 'a'], [5, 2, 'b'], [1, 2, 'a'], [1, 2, 'c'],
             [2, 0, 'd'], [2, 3, 'd'], [5, 2, 'd'], [1, 2, 'a'], [1, 2, 'd']]
        y = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
        atom = ATOMClassifier(X, y)
        atom.encode(max_onehot=4)
        self.assertTrue('Feature 2_c' in atom.dataset.columns)

    def test_target_encoder(self):
        ''' Test if the target-encoder works as intended '''

        X = [[2, 0, 'a'], [2, 3, 'a'], [5, 2, 'b'], [1, 2, 'a'], [1, 2, 'c'],
             [2, 0, 'd'], [2, 3, 'd'], [5, 2, 'd'], [1, 2, 'a'], [1, 2, 'd']]
        y = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
        atom = ATOMClassifier(X, y, random_state=1)
        atom.encode(max_onehot=None)
        self.assertEqual(len(atom.X.columns), len(X[0]))
        self.assertAlmostEqual(atom.dataset['Feature 2'][0], 2./3.)
        self.assertAlmostEqual(atom.dataset['Feature 2'][4], 0.5)


class outliers(unittest.TestCase):

    # << ================ Test parameters ================ >>

    def test_max_sigma_parameter(self):
        ''' Test if the max_sigma parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.outliers, max_sigma='test')
        self.assertRaises(ValueError, atom.outliers, max_sigma=0)

    def test_include_target_parameter(self):
        ''' Test if the include_target parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.outliers, include_target=1.1)

    # << ================ Test functionality ================ >>

    def test_remove_outliers(self):
        ''' Test if method works as intended '''

        X = [[0.21, 2.1, 1], [23, 2, 1], [0.2, 2.1, 1], [0.24, 2, 1],
             [0.23, 2, 2], [0.19, 0, 1], [0.21, 2, 2], [0.2, 2, 1],
             [0.2, 2, 1], [0.2, 2, 0]]
        y = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1]
        atom = ATOMClassifier(X, y, test_size=0.1, random_state=1)
        length = len(atom.train)
        atom.outliers(max_sigma=2)
        self.assertEqual(len(atom.train) + 1, length)

    def test_remove_outlier_in_target(self):
        ''' Test if method works as intended for target columns as well '''

        X = [[0.2, 2, 1], [0.2, 2, 1], [0.2, 2, 2], [0.24, 2, 1], [0.23, 2, 2],
             [0.19, 0, 1], [0.21, 3, 2], [0.2, 2, 1], [0.2, 2, 1], [0.2, 2, 0]]
        y = [0, 1, 0, 1, 1, 0, 245, 0, 1, 1]
        atom = ATOMClassifier(X, y, test_size=0.1, random_state=1)
        length = len(atom.train)
        atom.outliers(max_sigma=2, include_target=True)
        self.assertEqual(len(atom.train) + 1, length)


class balance(unittest.TestCase):

    # << ================ Test parameters ================ >>

    def test_not_classification_task(self):
        ''' Test if error s raised when task == regression '''

        X, y = load_boston(return_X_y=True)
        atom = ATOMRegressor(X, y)
        self.assertRaises(ValueError, atom.balance, undersample=0.8)

    def test_oversample_parameter(self):
        ''' Test if the oversample parameter is set correctly '''

        # Binary classification tasks
        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.balance, oversample=False)
        self.assertRaises(TypeError, atom.balance, oversample=0)
        self.assertRaises(ValueError, atom.balance, oversample=-2.1)
        self.assertRaises(ValueError, atom.balance, oversample='test')

        # Multiclass classification tasks
        X, y = load_wine(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.balance, undersample=1.0)

    def test_undersample_parameter(self):
        ''' Test if the undersample parameter is set correctly '''

        # Binary classification tasks
        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.balance, undersample=[2, 2])
        self.assertRaises(ValueError, atom.balance, undersample=-3.)
        self.assertRaises(ValueError, atom.balance, undersample='test')

        # Multiclass classification tasks
        X, y = load_wine(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.balance, undersample=0.8)

    def test_n_neighbors_parameter(self):
        ''' Test if the n_neighbors parameter is set correctly '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(TypeError, atom.balance, n_neighbors=2.2)
        self.assertRaises(ValueError, atom.balance, n_neighbors=0)

    def test_None_both_parameter(self):
        ''' Test if error raises when over and undersample are both None '''

        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y)
        self.assertRaises(ValueError, atom.balance)

    # << ================ Test functionality ================ >>

    def test_oversampling_method(self):
        ''' Test if the oversampling method works as intended '''

        # Binary classification (1 is majority class)
        strats = [1.0, 0.9, 'minority', 'not majority', 'all']
        for strat in strats:
            X, y = load_breast_cancer(return_X_y=True)
            atom = ATOMClassifier(X, y, random_state=1)
            length = (atom.y_train == 0).sum()
            atom.balance(oversample=strat)
            self.assertNotEqual((atom.y_train == 0).sum(), length)

        # Multiclass classification
        strats = ['minority', 'not majority', 'all']
        for strat in strats:
            X, y = load_wine(return_X_y=True)
            atom = ATOMClassifier(X, y, random_state=1)
            length = (atom.y_train == 2).sum()
            atom.balance(oversample=strat)
            self.assertNotEqual((atom.y_train == 2).sum(), length)

    def test_undersampling_method(self):
        ''' Test if the undersampling method works as intended '''

        # Binary classification (1 is majority class)
        strats = [1.0, 0.7, 'majority', 'not minority', 'all']
        for strat in strats:
            X, y = load_breast_cancer(return_X_y=True)
            atom = ATOMClassifier(X, y, random_state=1)
            length = (atom.y_train == 1).sum()
            atom.balance(undersample=strat)
            self.assertNotEqual((atom.y_train == 1).sum(), length)

        # Multiclass classification
        strats = ['majority', 'not minority', 'all']
        for strat in strats:
            X, y = load_wine(return_X_y=True)
            atom = ATOMClassifier(X, y, random_state=1)
            length = (atom.y_train == 1).sum()
            atom.balance(undersample=strat)
            self.assertNotEqual((atom.y_train == 1).sum(), length)


if __name__ == '__main__':

    unittest.main()

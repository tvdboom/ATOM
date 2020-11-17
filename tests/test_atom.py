# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for atom.py

"""

# Standard packages
import glob
import pytest
import numpy as np
import pandas as pd

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.utils import merge, check_scaling
from .utils import (
    FILE_DIR, X_bin, y_bin, X_class, y_class, X_reg, y_reg, X_bin_array,
    y_bin_array, X10, X10_nan, X10_str, y10, y10_nan, y10_str, y10_sn
)


# Test __init__ ==================================================== >>

def test_input_is_prepared():
    """Assert that the _prepare_input method form BaseTransformer is run."""
    atom = ATOMClassifier(X_bin_array, y_bin_array, random_state=1)
    assert isinstance(atom.X, pd.DataFrame)
    assert isinstance(atom.y, pd.Series)


def test_merger_to_dataset():
    """Assert that the merger between X and y was successful."""
    # Reset index since order of rows is different after shuffling
    merger = X_bin.merge(y_bin.to_frame(), left_index=True, right_index=True)
    df1 = merger.sort_values(by=merger.columns.tolist())
    df1.reset_index(drop=True, inplace=True)

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    df2 = atom.dataset.sort_values(by=atom.dataset.columns.tolist())
    df2.reset_index(drop=True, inplace=True)
    assert df1.equals(df2)


def test_test_size_attribute():
    """Assert that the _test_size attribute is created."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, random_state=1)
    assert atom._test_size == len(atom.test) / len(atom.dataset)


def test_raise_one_target_value():
    """Assert that error raises when there is only 1 target value."""
    y = [1 for _ in range(len(y_bin))]  # All targets are equal to 1
    pytest.raises(ValueError, ATOMClassifier, X_bin, y, random_state=1)


def test_mapping_assignment():
    """Assert that the mapping attribute is created."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.mapping == {'0': 0, '1': 1}


def test_mapping_with_nans():
    """Assert that the mapping attribute is created when str and nans are mixed."""
    atom = ATOMClassifier(X10, y10_sn, random_state=1)
    assert atom.mapping == {'n': 'n', 'nan': np.NaN, 'y': 'y'}


def test_task_assigning():
    """Assert that the task attribute is assigned correctly."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.task == "binary classification"

    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.task == "multiclass classification"

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    assert atom.task == "regression"


# Test __repr__ ==================================================== >>

def test_repr():
    """Assert that the __repr__ method visualizes the pipeline(s)."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    assert str(atom).startswith("ATOMClassifier")


# Test branch properties =========================================== >>

def test_branch_setter_invalid():
    """Assert that an error is raised when the name is empty."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    with pytest.raises(ValueError, match=r".*Can't create a branch.*"):
        atom.branch = ""


def test_branch_setter_change():
    """Assert that we can change to an old branch."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.branch = "branch_2"
    atom.clean()
    atom.branch = "main"
    assert atom.branch.estimators.empty  # Has no clean estimator


def test_branch_setter_new():
    """Assert that we can create a new pipeline."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.clean()
    atom.branch = "branch_2"
    assert list(atom._branches.keys()) == ["main", "branch_2"]


def test_branch_deleter():
    """Assert that we can delete a branch."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.branch = "branch_2"
    del atom.branch
    assert list(atom._branches.keys()) == ["main"]


# Test utility properties ========================================== >>

def test_nans():
    """Assert that nans returns a series of missing values."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    assert atom.nans.sum() == 1


def test_n_nans():
    """Assert that n_nans returns the number of missing values."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    assert atom.n_nans == 1


def test_categorical():
    """Assert that categorical returns a list of categorical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.categorical == ["Feature 2"]


def test_n_categorical():
    """Assert that n_categorical returns the number of categorical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.categorical == ["Feature 2"]


def test_scaled():
    """Assert that scaled returns if the dataset is scaled."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert not atom.scaled
    atom.scale()
    assert atom.scaled


# Test report ====================================================== >>

def test_creates_report():
    """Assert that the report attribute and file are created."""
    atom = ATOMClassifier(X_reg, y_reg, random_state=1)
    atom.report(n_rows=10, filename=FILE_DIR + "report")
    assert glob.glob(FILE_DIR + "report.html")


# Test transform =================================================== >>

def test_transform_method():
    """ Assert that the transform method works as intended """
    atom = ATOMClassifier(X10_str, y10_str, random_state=1)
    atom.clean()
    atom.encode(max_onehot=None)
    atom.run("Tree")

    # With default arguments
    X_trans = atom.transform(X10_str)
    assert X_trans["Feature 2"].dtype.kind in "ifu"

    # Changing arguments
    X_trans = atom.transform(X10_str, encode=False)
    assert X_trans["Feature 2"].dtype.kind not in "ifu"


def test_verbose_raises_when_invalid():
    """Assert an error is raised for an invalid value of verbose."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.transform, X_bin, verbose=3)


def test_verbose_in_transform():
    """Assert that the verbosity of the transformed classes is changed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    _ = atom.transform(X_bin, verbose=2)
    assert atom.branch.estimators[0].verbose == 2


def test_pipeline_parameter():
    """Assert that the pipeline parameter is obeyed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.outliers(max_sigma=1)
    X = atom.transform(X_bin, pipeline=[0])  # Only use Cleaner
    assert len(X) == len(X_bin)


def test_default_parameters():
    """Assert that outliers and balance are False by default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance()
    X = atom.transform(X_bin)
    assert len(X) == len(X_bin)


def test_parameters_are_obeyed():
    """Assert that only the transformations for the selected parameters are done."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.outliers(max_sigma=1)
    X = atom.transform(X_bin, outliers=True)
    assert len(X) != len(X_bin)


def test_transform_with_y():
    """Assert that the transform method works when y is provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.outliers(max_sigma=2, include_target=True)
    X, y = atom.transform(X_bin, y_bin, outliers=True)
    assert len(y) < len(y_bin)


# Test save_data =================================================== >>

def test_save_data():
    """Assert that the dataset is saved to a csv file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save_data(FILE_DIR + 'dataset')
    assert glob.glob(FILE_DIR + 'dataset')


# Test data cleaning methods ======================================= >>

def test_ATOM_params_to_method():
    """Assert that the ATOM parameters are passed to the method."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.scale()
    assert atom.branch.estimators[0].verbose == 1


def test_custom_params_to_method():
    """Assert that a custom parameter is passed to the method."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.scale(verbose=2)
    assert atom.branch.estimators[0].verbose == 2


def test_scale():
    """Assert that the scale method normalizes the features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert check_scaling(atom.dataset)


def test_clean():
    """Assert that the clean method cleans the dataset."""
    atom = ATOMClassifier(X10, y10_nan, random_state=1)
    atom.clean()
    assert len(atom.dataset) == 9


def test_impute():
    """Assert that the impute method imputes all missing values."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.impute()
    assert atom.dataset.isna().sum().sum() == 0


def test_encode():
    """Assert that the encode method encodes all categorical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    assert all([atom.X[col].dtype.kind in "ifu" for col in atom.X.columns])


def test_outliers():
    """Assert that the outliers method handles outliers in the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    length = len(atom.train)
    atom.outliers()
    assert len(atom.train) != length


def test_balance_wrong_task():
    """Assert that an error is raised for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(PermissionError, atom.balance, oversample=0.7)


def test_balance():
    """Assert that the balance method balances the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    length = (atom.y_train == 0).sum()
    atom.balance()
    assert (atom.y_train == 0).sum() != length


def test_balance_mapping():
    """Assert that the balance method gets the mapping attribute from ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance()
    assert atom.branch.estimators[0].mapping == atom.branch.mapping


def test_balance_attribute():
    """Assert that Balancer's estimator is attached to ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance(strategy="NearMiss")
    assert atom.nearmiss.__class__.__name__ == "NearMiss"


# Test feature engineering methods ================================= >>

def test_feature_generation():
    """Assert that the feature_generation method creates extra features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_generation(n_features=2, generations=5, population=200)
    assert atom.X.shape[1] == X_bin.shape[1] + 2


def test_feature_generation_attributes():
    """Assert that the attrs from feature_generation are passed to ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_generation("GFG", n_features=2, generations=5, population=200)
    assert hasattr(atom, "symbolic_transformer")
    assert hasattr(atom, "genetic_features")


def test_feature_selection_attrs():
    """Assert that feature_selection attaches only used attributes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="pca", n_features=8, max_correlation=0.8)
    assert hasattr(atom, "collinear") and hasattr(atom, "pca")


def test_default_solver_univariate():
    """Assert that the default solver is selected for strategy="univariate"."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="univariate", solver=None, n_features=8)
    assert atom.branch.estimators[0].solver.__name__ == "f_classif"

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy="univariate", solver=None, n_features=8)
    assert atom.branch.estimators[0].solver.__name__ == "f_regression"


def test_winner_solver_after_run():
    """Assert that the solver is the winning model after run."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("LR")
    atom.feature_selection(
        strategy="SFM", solver=None, n_features=8, max_correlation=None
    )
    assert atom.branch.estimators[0].solver is atom.winner.estimator


def test_default_solver_from_task():
    """Assert that the solver is inferred from the task when a model is selected."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="rfe", solver="lgb", n_features=8)
    assert type(atom.branch.estimators[0].solver).__name__ == "LGBMClassifier"

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy="rfe", solver="lgb", n_features=8)
    assert type(atom.branch.estimators[0].solver).__name__ == "LGBMRegressor"


def test_default_scoring_RFECV():
    """Assert that the scoring for RFECV is ATOM's metric_ when exists."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="recall")
    atom.feature_selection(strategy="rfecv", solver="lgb", n_features=8)
    assert atom.branch.estimators[0].kwargs["scoring"].name == "recall"


def test_plot_methods_attached():
    """Assert that the plot methods are attached to the ATOM instance."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="rfecv", solver="lgb", n_features=8)
    assert hasattr(atom, "plot_rfecv")


# Test training methods ============================================ >>

def test_errors_are_passed_to_ATOM():
    """Assert that the errors found in models are passed to ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"], n_calls=5, n_initial_points=(2, -1))
    assert atom.errors.get("LGB")


def test_models_and_metric_are_updated():
    """Assert that the models and metric attributes are updated correctly."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["LGB", "CatB"], metric="max_error")
    assert atom.models == ["LGB", "CatB"]
    assert atom.metric == "max_error"


def test_results_are_attached():
    """Assert that the results are attached for subsequent runs of Trainer."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("OLS")
    atom.run("LGB")
    assert len(atom.results) == 2


def test_errors_are_removed():
    """Assert that the errors are removed if subsequent runs are successful."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    # Invalid dimensions to create an error
    atom.run(["BNB", "LGB"], bo_params={"dimensions": {"LGB": 2}})
    atom.run("LGB")  # Runs correctly
    assert not atom.errors  # Errors should be empty


def test_model_subclasses_are_attached():
    """Assert that the model subclasses are attached to ATOM."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("OLS")
    assert hasattr(atom, "OLS") and hasattr(atom, "ols")


def test_branch_estimators_are_attached():
    """Assert that the branch estimators are attached to the models."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("OLS")
    assert hasattr(atom.ols, "_est_branch")


def test_errors_are_updated():
    """Assert that the found exceptions are updated in the errors attribute."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["XGB", "LGB"], n_calls=2, n_initial_points=2)  # No errors
    atom.run(["XGB", "LGB"], n_calls=2, n_initial_points=(2, -1))  # Produces an error
    assert atom.errors.get("LGB")


def test_exception_mixed_approaches():
    """Assert that an exception is raised when approaches are mixed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving(["lr", "lda"])
    pytest.raises(PermissionError, atom.run, "lr")


def test_assign_existing_metric():
    """Assert that the existing metric_ is assigned if rerun."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="recall")
    atom.run("lda")
    assert atom.metric == "recall"


def test_raises_invalid_metric_consecutive_runs():
    """Assert that an error is raised for a different metric_."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="recall")
    pytest.raises(ValueError, atom.run, "lda", metric="f1")


# Test property getters ============================================ >>

def test_dataset_setter():
    """Assert that the dataset setter changes the whole dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.dataset = merge(X_class, y_class)
    assert atom.dataset.shape == (len(X_class), X_class.shape[1] + 1)


def test_train_setter():
    """Assert that the train setter changes the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train = atom.train.iloc[:100, :]
    assert atom.train.shape == (100, X_bin.shape[1] + 1)


def test_test_setter():
    """Assert that the test setter changes the test set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.test = atom.test.iloc[:100, :]
    assert atom.test.shape == (100, X_bin.shape[1] + 1)


def test_X_setter():
    """Assert that the X setter changes the feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.X = atom.X.iloc[:, :10]
    assert atom.X.shape == (len(X_bin), 10)


def test_y_setter():
    """Assert that the y setter changes the target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y[0] == 1  # First value is 1 in original
    atom.y = [0] + list(y_bin.values[1:])
    assert atom.y[0] == 0  # First value changed to 0


def test_X_train_setter():
    """Assert that the X_train setter changes the training feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X_train = atom.X_train
    new_X_train.iloc[0, 0] = 999
    atom.X_train = new_X_train
    assert atom.X_train.iloc[0, 0] == 999


def test_X_test_setter():
    """Assert that the X_test setter changes the test feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X_test = atom.X_test
    new_X_test.iloc[0, 0] = 999
    atom.X_test = new_X_test
    assert atom.X_test.iloc[0, 0] == 999


def test_y_train_setter():
    """Assert that the y_train setter changes the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y_train.iloc[0] == 1  # First value is 1 in original
    atom.y_train = [0] + list(atom.y_train.values[1:])
    assert atom.y_train.iloc[0] == 0  # First value changed to 0


def test_y_test_setter():
    """Assert that the y_test setter changes the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y_test.iloc[0] == 0  # First value is 0 in original
    atom.y_test = [1] + list(atom.y_test.values[1:])
    assert atom.y_test.iloc[0] == 1  # First value changed to 1


def test_data_properties_to_df():
    """Assert that the data attributes are converted to a df at setter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.X = X_bin_array
    assert isinstance(atom.X, pd.DataFrame)


def test_data_properties_to_series():
    """Assert that the data attributes are converted to a series at setter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.y = y_bin_array
    assert isinstance(atom.y, pd.Series)


def test_setter_error_unequal_rows():
    """Assert that an error is raised when the setter has unequal rows."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=r"number of rows"):
        atom.X_train = X_bin


def test_setter_error_unequal_index():
    """Assert that an error is raised when the setter has unequal indices."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=r"the same indices"):
        atom.y = pd.Series(y_bin_array, index=range(10, len(y_bin_array) + 10))


def test_setter_error_unequal_columns():
    """Assert that an error is raised when the setter has unequal columns."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match="number of columns"):
        new_X = atom.train
        new_X.insert(0, "new_column", 1)
        atom.train = new_X


def test_setter_error_unequal_column_names():
    """Assert that an error is raised with different column names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match="the same columns"):
        new_X = atom.train.drop(atom.train.columns[0], axis=1)
        new_X.insert(0, "new_column", 1)
        atom.train = new_X


def test_setter_error_unequal_target_names():
    """Assert that an error is raised with different target names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match="the same name"):
        new_y_train = atom.y_train
        new_y_train.name = "different_name"
        atom.y_train = new_y_train

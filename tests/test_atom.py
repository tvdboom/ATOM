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
from atom.utils import check_scaling
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

def test_branch_setter_empty():
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
    assert atom.pipeline.empty  # Has no clean estimator


def test_branch_setter_new():
    """Assert that we can create a new pipeline."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.clean()
    atom.branch = "branch_2"
    assert list(atom._branches.keys()) == ["main", "branch_2"]


def test_branch_setter_from_valid():
    """Assert that we cna create a new pipeline not from the current one."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.branch = "branch_2"
    atom.impute()
    atom.branch = "branch_3_from_main"
    assert atom.n_nans > 0


def test_branch_setter_from_invalid():
    """Assert that an error is raised when the from branch doesn't exist."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    with pytest.raises(ValueError, match=r".*branch to split from does not exist.*"):
        atom.branch = "new_branch_from_invalid"


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
    assert atom.pipeline[0].verbose == 1


def test_custom_params_to_method():
    """Assert that a custom parameter is passed to the method."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.scale(verbose=2)
    assert atom.pipeline[0].verbose == 2


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
    assert atom.pipeline[0].mapping == atom.mapping


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
    atom.feature_selection(strategy="univariate", n_features=8)
    assert hasattr(atom, "univariate")
    assert not hasattr(atom, "RFE")


def test_default_solver_univariate():
    """Assert that the default solver is selected for strategy="univariate"."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="univariate", solver=None, n_features=8)
    assert atom.pipeline[0].solver.__name__ == "f_classif"

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy="univariate", solver=None, n_features=8)
    assert atom.pipeline[0].solver.__name__ == "f_regression"


def test_winner_solver_after_run():
    """Assert that the solver is the winning model after run."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("LR")
    atom.feature_selection(strategy="SFM", solver=None, n_features=8)
    assert atom.pipeline[0].solver is atom.winner.estimator


def test_default_solver_from_task():
    """Assert that the solver is inferred from the task when a model is selected."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="rfe", solver="lgb", n_features=8)
    assert type(atom.pipeline[0].solver).__name__ == "LGBMClassifier"

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy="rfe", solver="lgb", n_features=8)
    assert type(atom.pipeline[0].solver).__name__ == "LGBMRegressor"


def test_default_scoring_RFECV():
    """Assert that the scoring for RFECV is ATOM's metric_ when exists."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="recall")
    atom.feature_selection(strategy="rfecv", solver="lgb", n_features=8)
    assert atom.pipeline[0].kwargs["scoring"].name == "recall"


# Test training methods ============================================ >>

def test_errors_are_updated():
    """Assert that the found exceptions are updated in the errors attribute."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["XGB", "LGB"], n_calls=2, n_initial_points=(2, -1))  # Produces an error
    assert list(atom.errors.keys()) == ["LGB"]


def test_models_and_metric_are_updated():
    """Assert that the models and metric attributes are updated correctly."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["LGB", "CatB"], metric="max_error")
    assert atom.models == ["LGB", "CatB"]
    assert atom.metric == "max_error"


def test_results_are_attached():
    """Assert that the results are attached for subsequent runs of Trainer."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
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
    atom.run("Tree")
    assert hasattr(atom, "Tree") and hasattr(atom, "tree")


def test_trainer_becomes_atom():
    """Assert that the parent trainer is converted to atom."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    assert atom is atom.tree.T


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

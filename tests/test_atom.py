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
from unittest.mock import patch
from sklearn.metrics import get_scorer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.data_cleaning import Imputer, Pruner
from atom.utils import check_scaling
from .utils import (
    FILE_DIR, X_bin, y_bin, X_class, y_class, X_reg, y_reg, X10,
    X10_nan, X10_str, y10, y10_str, y10_sn, X20_out,
)


# Test __init__ ==================================================== >>

def test_test_size_attribute():
    """Assert that the _test_size attribute is created."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, n_jobs=2, random_state=1)
    assert atom._test_size == len(atom.test) / len(atom.dataset)


def test_task_assignment():
    """Assert that the correct task is assigned."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.task == "binary classification"

    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.task == "multiclass classification"

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    assert atom.task == "regression"


def test_raise_one_target_value():
    """Assert that error raises when there is only 1 target value."""
    y = [1 for _ in range(len(y_bin))]  # All targets are equal to 1
    pytest.raises(ValueError, ATOMClassifier, X_bin, y, random_state=1)


def test_mapping_assignment():
    """Assert that the mapping attribute is created."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.mapping == {"0": 0, "1": 1}


def test_mapping_with_nans():
    """Assert that the mapping attribute is created when str and nans are mixed."""
    atom = ATOMClassifier(X10, y10_sn, random_state=1)
    assert atom.mapping == {"n": 'n', "nan": np.NaN, "y": 'y'}


# Test magic methods =============================================== >>

def test_repr():
    """Assert that the __repr__ method visualizes the pipeline(s)."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert "Branches: master" in str(atom)
    atom.branch = "branch_2"
    assert "Branches:\n   >>> master\n   >>> branch_2 !" in str(atom)


def test_iter():
    """Assert that we can iterate over atom."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    assert [item for item in atom][1] == ("imputer", atom.pipeline[1])


def test_len():
    """Assert that the length of atom is the length of the pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    assert len(atom) == 2


def test_getitem():
    """Assert that atom is subscriptable."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    atom.encode()
    assert len(atom[1:3]) == 2
    assert isinstance(atom["imputer"], Imputer)
    assert isinstance(atom[1], Imputer)


# Test utility properties =========================================== >>

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
    atom.branch = "master"
    assert atom.pipeline.empty  # Has no clean estimator


def test_branch_setter_new():
    """Assert that we can create a new pipeline."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.clean()
    atom.branch = "branch_2"
    assert list(atom._branches.keys()) == ["master", "branch_2"]


def test_branch_setter_from_valid():
    """Assert that we cna create a new pipeline not from the current one."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.branch = "branch_2"
    atom.impute()
    atom.branch = "branch_3_from_master"
    assert atom.n_nans > 0


def test_branch_setter_from_invalid():
    """Assert that an error is raised when the from branch doesn't exist."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    with pytest.raises(ValueError, match=r".*branch to split from does not exist.*"):
        atom.branch = "new_branch_from_invalid"


def test_scaled():
    """Assert that scaled returns if the dataset is scaled."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert not atom.scaled
    atom.scale()
    assert atom.scaled


def test_duplicates():
    """Assert that duplicates returns the number of duplicated samples."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    assert atom.duplicates == 2


def test_nans():
    """Assert that nans returns a series of missing values."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    assert atom.nans.sum() == 2


def test_n_nans():
    """Assert that n_nans returns the number of rows with missing values."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    assert atom.n_nans == 2


def test_numerical():
    """Assert that numerical returns the names of the numerical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.numerical == ["Feature 1", "Feature 2"]


def test_n_numerical():
    """Assert that n_categorical returns the number of numerical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.n_numerical == 2


def test_categorical():
    """Assert that categorical returns the names of categorical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.categorical == ["Feature 3"]


def test_n_categorical():
    """Assert that n_categorical returns the number of categorical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.n_categorical == 1


def test_outliers():
    """Assert that nans returns a series of outlier values."""
    atom = ATOMClassifier(X20_out, y10 * 2, random_state=1)
    assert atom.outliers.sum() == 2


def test_n_outliers():
    """Assert that n_outliers returns the number of rows with outliers."""
    atom = ATOMClassifier(X20_out, y10 * 2, random_state=1)
    assert atom.n_outliers == 2


def test_classes_property():
    """Assert that the classes property returns a df of the classes in y."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert list(atom.classes.index) == [0, 1, 2]


def test_n_classes_property():
    """Assert that the n_classes property returns the number of classes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.n_classes == 3


# Test utility methods ============================================= >>

@pytest.mark.parametrize("column", ["Feature 1", 1])
def test_distribution(column):
    """Assert that the distribution method and file are created."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    pytest.raises(ValueError, atom.distribution, column="Feature 3")
    df = atom.distribution(column=column)
    assert len(df) == 11


@patch("atom.atom.ProfileReport")
def test_report(cls):
    """Assert that the report method and file are created."""
    atom = ATOMClassifier(X_reg, y_reg, random_state=1)
    atom.report(n_rows=10, filename="report")
    cls.return_value.to_file.assert_called_once_with("report.html")


def test_transform_method():
    """ Assert that the transform method works as intended."""
    atom = ATOMClassifier(X10_str, y10_str, random_state=1)
    atom.clean()
    atom.encode(max_onehot=None)
    atom.run("Tree")

    # With default arguments
    X_trans = atom.transform(X10_str)
    assert X_trans["Feature 3"].dtype.kind in "ifu"

    # Changing arguments
    X_trans = atom.transform(X10_str, encoder=False)
    assert X_trans["Feature 3"].dtype.kind not in "ifu"


def test_verbose_raises_when_invalid():
    """Assert an error is raised for an invalid value of verbose."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    pytest.raises(ValueError, atom.transform, X_bin, verbose=3)


def test_pipeline_parameter():
    """Assert that the pipeline parameter is obeyed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.prune(max_sigma=1)
    X = atom.transform(X_bin, pipeline=[0])  # Only use Cleaner
    assert len(X) == len(X_bin)


def test_default_parameters():
    """Assert that prune and balance are False by default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance()
    X = atom.transform(X_bin)
    assert len(X) == len(X_bin)


def test_parameters_are_obeyed():
    """Assert that it only transforms for the selected parameters."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.prune(max_sigma=1)
    X = atom.transform(X_bin, pruner=True)
    assert len(X) != len(X_bin)


def test_transform_with_y():
    """Assert that the transform method works when y is provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.prune(strategy="iforest", include_target=True)
    X, y = atom.transform(X_bin, y_bin, pruner=True)
    assert len(y) < len(y_bin)


def test_save_data():
    """Assert that the dataset is saved to a csv file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save_data(FILE_DIR + "auto")
    assert glob.glob(FILE_DIR + "ATOMClassifier_dataset.csv")


def test_export_pipeline():
    """Assert that we can export the pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(PermissionError, atom.export_pipeline)
    atom.clean()
    atom.run("GNB")
    assert len(atom.export_pipeline("GNB")) == 2  # Without scaler
    atom.run("LGB")
    assert len(atom.export_pipeline("LGB")) == 3  # With StandardScaler
    atom.scale()
    assert len(atom.export_pipeline("LGB")) == 3  # With Scaler


# Test transformer methods ========================================= >>

def test_params_to_method():
    """Assert that atom's parameters are passed to the method."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.scale()
    assert atom.pipeline[0].verbose == 1


def test_custom_params_to_method():
    """Assert that a custom parameter is passed to the method."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.scale(verbose=2)
    assert atom.pipeline[0].verbose == 2


def test_add_pipeline():
    """Assert that adding a pipeline adds every individual step."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("sfm", SelectFromModel(RandomForestClassifier())),
        ]
    )
    atom = ATOMClassifier(X_bin, y_bin, verbose=2, random_state=1)
    atom.add(pipeline)
    assert list(atom.pipeline.index) == ["scaler", "sfm"]


def test_no_transformer():
    """Assert that an error is raised if the estimator is not a transformer."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.add, RandomForestClassifier)


def test_basetransformer_params_are_attached():
    """Assert that the n_jobs and random_state params from atom are used."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(PCA())  # When left to default
    atom.add(PCA(random_state=2))  # When changed
    assert atom.pipeline[0].get_params()["random_state"] == 1
    assert atom.pipeline[1].get_params()["random_state"] == 2


def test_add_train_only():
    """Assert that atom accepts custom transformers for the train set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(StandardScaler(), train_only=True)
    assert check_scaling(atom.X_train) and not check_scaling(atom.X_test)

    len_train, len_test = len(atom.train), len(atom.test)
    atom.add(Pruner(), train_only=True)
    assert len(atom.train) != len_train and len(atom.test) == len_test


def test_add_complete_dataset():
    """Assert that atom accepts custom transformers for the complete dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(StandardScaler())
    assert check_scaling(atom.dataset)

    len_dataset = len(atom.dataset)
    atom.add(Pruner())
    assert len(atom.dataset) != len_dataset


def test_keep_column_names():
    """Assert that the column names are kept after transforming."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # When the columns are only transformed
    cols = atom.columns.copy()
    atom.add(StandardScaler())
    assert atom.columns == cols

    # When columns were removed
    cols = atom.columns.copy()
    atom.add(SelectFromModel(RandomForestClassifier()))
    assert all(col in cols for col in atom.columns)


def test_subset_columns():
    """Assert that you can use a subset of the columns."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # Column indices
    cols = atom.columns.copy()
    atom.scale(columns=[3, 4])
    assert atom.columns == cols  # All columns are kept
    assert check_scaling(atom.X.iloc[:, [3, 4]])
    assert not check_scaling(atom.dataset.iloc[:, [7, 8]])

    # Column names
    atom.scale(columns=["mean radius", "mean texture"])
    assert check_scaling(atom.dataset.iloc[:, [0, 1]])

    # Column slice
    atom.scale(columns=slice(10, 12))
    assert check_scaling(atom.dataset.iloc[:, [10, 11]])


def test_sets_are_kept_equal():
    """Assert that the train and test sets always keep the same rows."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    len_train, len_test = len(atom.train), len(atom.test)
    atom.add(Pruner())
    assert len(atom.train) < len_train and len(atom.test) < len_test


def test_scale():
    """Assert that the scale method normalizes the features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert check_scaling(atom.dataset)


def test_clean():
    """Assert that the clean method cleans the dataset."""
    atom = ATOMClassifier(X10, y10_sn, random_state=1)
    atom.clean()
    assert len(atom.dataset) == 9
    assert atom.mapping == {"n": 0, "y": 1}


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


def test_prune():
    """Assert that the prune method handles outliers in the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    len_train, len_test = len(atom.train), len(atom.test)
    atom.prune()
    assert len(atom.train) != len_train and len(atom.test) == len_test


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
    """Assert that the balance method gets the mapping attribute from atom."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance()
    assert atom.pipeline[0].mapping == atom.mapping


def test_balance_attribute():
    """Assert that Balancer's estimator is attached to ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance(strategy="NearMiss")
    assert atom.nearmiss.__class__.__name__ == "NearMiss"


def test_feature_generation():
    """Assert that the feature_generation method creates extra features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_generation(n_features=2, generations=5, population=200)
    assert atom.X.shape[1] == X_bin.shape[1] + 2


def test_feature_generation_attributes():
    """Assert that the attrs from feature_generation are passed to atom."""
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
    atom.feature_selection(strategy="rfe", solver="lgb", n_features=25)
    assert type(atom.pipeline[0].solver).__name__ == "LGBMRegressor"


@patch("atom.feature_engineering.SequentialFeatureSelector")
def test_default_scoring(cls):
    """Assert that the scoring is atom's metric when exists."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="recall")
    atom.feature_selection(strategy="sfs", solver="lgb", n_features=25)
    assert atom.pipeline[0].kwargs["scoring"].name == "recall"


@patch("tpot.TPOTClassifier")
def test_automl_classification(cls):
    """Assert that the automl method works for classification tasks."""
    pl = Pipeline(
        steps=[
            ('standardscaler', StandardScaler()),
            ('robustscaler', RobustScaler()),
            ('mlpclassifier', MLPClassifier(alpha=0.001, random_state=1))
        ]
    )
    cls.return_value.fitted_pipeline_ = pl.fit(X_bin, y_bin)

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric="accuracy")
    atom.automl()
    cls.assert_called_with(
        n_jobs=1,
        random_state=1,
        scoring=atom._metric[0],  # Called using atom's metric
        verbosity=0
    )
    assert len(atom) == 2
    assert atom.models == ["Tree", "MLP"]


@patch("tpot.TPOTRegressor")
def test_automl_regression(cls):
    """Assert that the automl method works for regression tasks."""
    pl = Pipeline(
        steps=[
            ('rbfsampler', RBFSampler(gamma=0.95, random_state=2)),
            ('lassolarscv', LassoLarsCV(normalize=False))
        ]
    )
    cls.return_value.fitted_pipeline_ = pl.fit(X_reg, y_reg)

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.automl(scoring="r2", random_state=2)
    cls.assert_called_with(n_jobs=1, scoring="r2", verbosity=0, random_state=2)
    assert atom.metric == "r2"


def test_invalid_scoring():
    """Assert that an error is raised when the provided scoring is invalid."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree", metric="mse")
    pytest.raises(ValueError, atom.automl, scoring="r2")


# Test training methods ============================================ >>

def test_assign_existing_metric():
    """Assert that the existing metric_ is assigned if rerun."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", metric="recall")
    atom.run("Tree")
    assert atom.metric == "recall"


def test_raises_invalid_metric_consecutive_runs():
    """Assert that an error is raised for a different metric."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", metric="recall")
    pytest.raises(ValueError, atom.run, "Tree", metric="f1")


def test_scaling_is_passed():
    """Assert that the scaling is passed to the trainer."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.scale("minmax")
    atom.run("LGB")
    assert atom.dataset.equals(atom.lgb.dataset)


def test_errors_are_updated():
    """Assert that the found exceptions are updated in the errors attribute."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)

    # Produce an error on one model (when n_initial_points > n_calls)
    atom.run(["Tree", "LGB"], n_calls=(3, 2), n_initial_points=(2, 5))
    assert list(atom.errors) == ["LGB"]

    # Subsequent runs should remove the original model
    atom.run(["Tree", "LGB"], n_calls=(5, 3), n_initial_points=(7, 1))
    assert atom.models == "LGB"


def test_models_and_metric_are_updated():
    """Assert that the models and metric attributes are updated correctly."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["OLS", "Tree"], metric=get_scorer("max_error"))
    assert atom.models == ["OLS", "Tree"]
    assert atom.metric == "max_error"


def test_errors_are_removed():
    """Assert that the errors are removed if subsequent runs are successful."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["BNB", "Tree"], bo_params={"dimensions": {"Tree": 2}})  # Invalid dims
    atom.run("Tree")  # Runs correctly
    assert not atom.errors  # Errors should be empty


def test_trainer_becomes_atom():
    """Assert that the parent trainer is converted to atom."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    assert atom is atom.tree.T

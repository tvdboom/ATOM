# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for atom.py

"""

import glob
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from category_encoders.leave_one_out import LeaveOneOutEncoder
from evalml.pipelines.components.estimators import SVMClassifier
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, MultiLabelBinarizer, OneHotEncoder, StandardScaler,
)

from atom import ATOMClassifier, ATOMRegressor
from atom.data_cleaning import Cleaner, Pruner
from atom.training import DirectClassifier
from atom.utils import check_scaling

from .conftest import (
    X10, DummyTransformer, X10_dt, X10_nan, X10_str, X10_str2, X20_out, X_bin,
    X_class, X_label, X_reg, X_sparse, X_text, merge, y10, y10_label,
    y10_label2, y10_sn, y10_str, y_bin, y_class, y_label, y_multiclass, y_reg,
)


# Test __init__ ==================================================== >>

def test_task_assignment():
    """Assert that the correct task is assigned."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.task == "binary classification"

    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.task == "multiclass classification"

    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    assert atom.task == "multilabel classification"

    atom = ATOMClassifier(X10, y=y10_label, stratify=False, random_state=1)
    assert atom.task == "multilabel classification"

    atom = ATOMClassifier(X10, y=y10_label2, stratify=False, random_state=1)
    assert atom.task == "multilabel classification"

    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    assert atom.task == "multiclass-multioutput classification"

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    assert atom.task == "regression"

    atom = ATOMRegressor(X_class, y=y_multiclass, random_state=1)
    assert atom.task == "multioutput regression"


def test_raise_one_target_value():
    """Assert that error raises when there is only 1 target value."""
    with pytest.raises(ValueError, match=".*1 target value.*"):
        ATOMClassifier(X_bin, [1] * len(X_bin), random_state=1)


def test_backend_with_n_jobs_1():
    """Assert that a warning is raised ."""
    with pytest.warns(UserWarning, match=".*Leaving n_jobs=1 ignores.*"):
        ATOMClassifier(X_bin, y_bin, warnings=True, backend="threading", random_state=1)


# Test magic methods =============================================== >>

def test_repr():
    """Assert that the __repr__ method visualizes the pipeline(s)."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert "Branches: master" in str(atom)
    atom.branch = "b2"
    assert "Branches:\n   --> master\n   --> b2 !" in str(atom)


def test_iter():
    """Assert that we can iterate over atom's pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    assert [item for item in atom][1] == atom.pipeline[1]


# Test utility properties =========================================== >>

def test_branch_same():
    """Assert that we can stay on the same branch."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    atom.branch = "master"
    assert atom.branch.name == "master"


def test_branch_change():
    """Assert that we can change to another branch."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    atom.branch = "b2"
    atom.clean()
    atom.branch = "master"
    assert atom.pipeline.empty  # Has no Cleaner


def test_branch_existing_name():
    """Assert that an error is raised when the name already exists."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    atom.branch = "b2"
    with pytest.raises(ValueError, match=".*already exists.*"):
        atom.branch = "b2_from_master"


def test_branch_unknown_parent():
    """Assert that an error is raised when the parent doesn't exist."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    with pytest.raises(ValueError, match=".*does not exist.*"):
        atom.branch = "b2_from_invalid"


def test_branch_new():
    """Assert that we can create a new branch."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    atom.clean()
    atom.branch = "b2"
    assert len(atom._branches) == 2


def test_branch_from_valid():
    """Assert that we can create a new branch, not from the current one."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.branch = "b2"
    atom.impute()
    atom.branch = "b3_from_master"
    assert atom.branch.name == "b3"
    assert atom.n_nans > 0


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
    assert len(atom.numerical) == 3


def test_n_numerical():
    """Assert that n_categorical returns the number of numerical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.n_numerical == 3


def test_categorical():
    """Assert that categorical returns the names of categorical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert len(atom.categorical) == 1


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


def test_classes():
    """Assert that the classes property returns a df of the classes in y."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert list(atom.classes.index) == [0, 1, 2]


def test_n_classes():
    """Assert that the n_classes property returns the number of classes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.n_classes == 3


# Test utility methods ============================================= >>

@patch("evalml.AutoMLSearch")
def test_automl(cls):
    """Assert that the automl method works."""
    pl = Pipeline([("scaler", StandardScaler()), ("clf", SVMClassifier())])
    cls.return_value.best_pipeline = pl.fit(X_bin, y_bin)

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric="accuracy")
    atom.branch = "automl"  # Change branch since new pipeline
    atom.automl()
    cls.assert_called_once()
    assert len(atom.pipeline) == 1
    assert atom.models == ["Tree", "SVM"]


@patch("evalml.AutoMLSearch")
def test_automl_custom_objective(cls):
    """Assert that the automl method works for a custom objective."""
    pl = Pipeline([("scaler", StandardScaler()), ("clf", SVMClassifier())])
    cls.return_value.best_pipeline = pl.fit(X_bin, y_bin)

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.automl(objective="r2")
    cls.assert_called_once()


def test_automl_invalid_objective():
    """Assert that an error is raised when the provided objective is invalid."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree", metric="mse")
    with pytest.raises(ValueError, match=".*objective parameter.*"):
        atom.automl(objective="r2")


@pytest.mark.parametrize("distributions", [None, "norm", ["norm", "pearson3"]])
def test_distribution(distributions):
    """Assert that the distribution method and file are created."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    df = atom.distribution(distributions=distributions, columns=(0, 1))
    assert isinstance(df, pd.DataFrame)


@patch("ydata_profiling.ProfileReport")
def test_eda(cls):
    """Assert that the eda method creates a report."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.eda(filename="report")
    cls.return_value.to_file.assert_called_once_with("report.html")


def test_load_no_atom():
    """Assert that an error is raised when the instance is not atom."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.save("trainer")
    with pytest.raises(ValueError, match=".*ATOMClassifier nor ATOMRegressor.*"):
        ATOMClassifier.load("trainer")


def test_load_already_contains_data():
    """Assert that an error is raised when data is provided without needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save("atom", save_data=True)
    with pytest.raises(ValueError, match=".*already contains data.*"):
        ATOMClassifier.load("atom", data=(X_bin,))


def test_load_with_data():
    """Assert that data can be loaded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save("atom", save_data=False)

    atom2 = ATOMClassifier.load("atom", data=(X_bin, y_bin))
    pd.testing.assert_frame_equal(atom2.dataset, atom.dataset, check_dtype=False)


def test_load_ignores_n_rows_parameter():
    """Assert that n_rows is not used when transform_data=False."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=0.6, random_state=1)
    atom.save("atom", save_data=False)

    atom2 = ATOMClassifier.load("atom", data=(X_bin, y_bin), transform_data=False)
    assert len(atom2.dataset) == len(X_bin)


def test_load_transform_data():
    """Assert that the data is transformed correctly."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale(columns=slice(3, 10))
    atom.apply(np.exp, columns=2)
    atom.feature_generation(strategy="dfs", n_features=5)
    atom.feature_selection(strategy="sfm", solver="lgb", n_features=10)
    atom.save("atom", save_data=False)

    atom2 = ATOMClassifier.load("atom", data=(X_bin, y_bin), transform_data=True)
    assert atom2.dataset.shape == atom.dataset.shape

    atom3 = ATOMClassifier.load("atom", data=(X_bin, y_bin), transform_data=False)
    assert atom3.dataset.shape == merge(X_bin, y_bin).shape


def test_load_transform_data_multiple_branches():
    """Assert that the data is transformed with multiple branches."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.prune()
    atom.branch = "b2"
    atom.balance()
    atom.feature_generation(strategy="dfs", n_features=5)
    atom.branch = "b3"
    atom.feature_selection(strategy="sfm", solver="lgb", n_features=20)
    atom.save("atom_2", save_data=False)

    atom2 = ATOMClassifier.load("atom_2", data=(X_bin, y_bin), transform_data=True)
    for branch in atom._branches:
        pd.testing.assert_frame_equal(
            left=atom2._branches[branch.name]._data,
            right=atom._branches[branch.name]._data,
            check_dtype=False,
        )


def test_inverse_transform():
    """Assert that the inverse_transform method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    atom.scale()
    atom.impute()  # Does nothing, but doesn't crash either
    pd.testing.assert_frame_equal(atom.inverse_transform(atom.X), X_bin)


def test_reset():
    """Assert that the reset method deletes models and branches."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.scale()
    atom.branch = "2"
    atom.encode()
    atom.run("LR")
    atom.reset()
    assert not atom.models and len(atom._branches) == 1
    assert atom["x2"].dtype.name == "object"  # Is reset back to str


def test_save_data():
    """Assert that the dataset is saved to a csv file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save_data("auto")
    assert glob.glob("ATOMClassifier_dataset.csv")


def test_shrink_dtypes_excluded():
    """Assert that some dtypes are excluded from changing."""
    atom = ATOMClassifier(X10_str2, y10, random_state=1)
    assert atom.dtypes[3].name == "bool"
    atom.shrink()
    assert atom.dtypes[3].name == "bool"


def test_shrink_obj2cat():
    """Assert that the obj2cat parameter works as intended."""
    atom = ATOMClassifier(X10_str2, y10, random_state=1)
    atom.shrink(obj2cat=False)
    assert atom.dtypes[2].name == "object"

    atom.shrink()
    assert atom.dtypes[2].name == "category"


def test_shrink_int2uint():
    """Assert that the int2uint parameter works as intended."""
    atom = ATOMClassifier(X10_str2, y10, random_state=1)
    assert atom.dtypes[0].name == "int64"
    atom.shrink()
    assert atom.dtypes[0].name == "int8"

    assert atom.dtypes[0].name == "int8"
    atom.shrink(int2uint=True)
    assert atom.dtypes[0].name == "uint8"


def test_shrink_sparse_arrays():
    """Assert that sparse arrays are also transformed."""
    atom = ATOMClassifier(X_sparse, y10, random_state=1)
    assert atom.dtypes[0].name == "Sparse[int64, 0]"
    atom.shrink()
    assert atom.dtypes[0].name == "Sparse[int8, 0]"


def test_shrink_dtypes_unchanged():
    """Assert that optimal dtypes are left unchanged."""
    atom = ATOMClassifier(X_bin.astype("float32"), y_bin, random_state=1)
    assert atom.dtypes[3].name == "float32"
    atom.shrink()
    assert atom.dtypes[3].name == "float32"


def test_shrink_dense2sparse():
    """Assert that the dataset can be converted to sparse."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.dtypes[0].name == "float64"
    atom.shrink(dense2sparse=True)
    assert atom.dtypes[0].name.startswith("Sparse[float32")


def test_shrink_exclude_columns():
    """Assert that columns can be excluded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.dtypes[0].name == "float64"
    assert atom.dtypes[-1].name != "int8"
    atom.shrink(columns=-1)
    assert atom.dtypes[0].name == "float64"
    assert atom.dtypes[-1].name == "int8"


def test_stats_mixed_sparse_dense():
    """Assert that stats show new information for mixed datasets."""
    X = X_sparse.copy()
    X["dense column"] = 2

    atom = ATOMClassifier(X, y10, random_state=1)
    atom.stats()


def test_status():
    """Assert that the status method prints an overview of the instance."""
    atom = ATOMClassifier(*make_classification(100000), random_state=1)
    atom.status()


def test_transform():
    """Assert that the transform method works as intended."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode(max_onehot=None)
    assert atom.transform(X10_str)["x2"].dtype.kind in "ifu"


def test_transform_not_train_only():
    """Assert that train_only transformers are not used."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.prune(max_sigma=2)
    assert len(atom.transform(X_bin)) == len(X_bin)


def test_transform_verbose_invalid():
    """Assert an error is raised for an invalid value of verbose."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    pytest.raises(ValueError, atom.transform, X_bin, verbose=3)


# Test base transformers =========================================== >>

def test_add_after_model():
    """Assert that an error is raised when adding after training a model."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.run("Dummy")
    with pytest.raises(PermissionError, match=".*not allowed to add transformers.*"):
        atom.scale()


def test_custom_params_to_method():
    """Assert that a custom parameter is passed to the method."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.scale(verbose=2)
    assert atom.pipeline[0].verbose == 2


def test_add_no_transformer():
    """Assert that an error is raised if the estimator has no estimator."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(AttributeError, match=".*should have a transform method.*"):
        atom.add(RandomForestClassifier())


def test_add_basetransformer_params_are_attached():
    """Assert that the n_jobs and random_state params from atom are used."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(PCA())  # When left to default
    atom.add(PCA(random_state=2))  # When changed
    assert atom.pipeline[0].get_params()["random_state"] == 1
    assert atom.pipeline[1].get_params()["random_state"] == 2


def test_add_train_only():
    """Assert that atom accepts transformers for the train set only."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(StandardScaler(), train_only=True)
    assert check_scaling(atom.X_train) and not check_scaling(atom.X_test)

    len_train, len_test = len(atom.train), len(atom.test)
    atom.add(Pruner(), train_only=True)
    assert len(atom.train) != len_train and len(atom.test) == len_test


def test_add_complete_dataset():
    """Assert that atom accepts transformers for the complete dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(StandardScaler())
    assert check_scaling(atom.dataset)

    len_dataset = len(atom.dataset)
    atom.add(Pruner())
    assert len(atom.dataset) != len_dataset


def test_add_transformer_only_y():
    """Assert that atom accepts transformers with only y."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    atom.add(LabelEncoder())
    assert np.all((atom["target"] == 0) | (atom["target"] == 1))


def test_add_transformer_y_ignore_X():
    """Assert that atom accepts transformers with y and default X."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    atom.clean()  # Cleaner has X=None and y=None
    y = atom.transform(y=y10_str)
    assert np.all((y == 0) | (y == 1))


def test_add_default_X_is_used():
    """Assert that X is autofilled when required but not provided."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    atom.clean(columns=-1)
    assert atom.mapping


def test_add_invalid_columns_only_y():
    """Assert that an error is raised when the transformer requires features."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    with pytest.raises(ValueError, match=".*trying to fit transformer.*"):
        atom.encode(columns=-1)  # Encoder.fit requires X


def test_returned_column_already_exists():
    """Assert that an error is raised if an existing column is returned."""

    def func_test(df):
        df["mean texture"] = 1
        return df

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*already exists in the original.*"):
        atom.apply(func_test, columns="!mean texture")


def test_add_sparse_matrices():
    """Assert that transformers that return sp.matrix are accepted."""
    atom = ATOMClassifier(X10_str, y10, shuffle=False, random_state=1)
    atom.add(OneHotEncoder(handle_unknown="ignore"), columns=2)
    assert atom.shape == (10, 8)  # Creates 4 extra columns


def test_add_keep_column_names():
    """Assert that the column names are kept after transforming."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)

    # Transformer has method get_feature_names_out
    atom.add(StandardScaler())
    assert atom.features.tolist() == ["x0", "x1", "x2", "x3"]

    # Transformer has method get_feature_names
    atom.add(LeaveOneOutEncoder(return_df=False))
    assert atom.features.tolist() == ["x0", "x1", "x2", "x3"]

    # Transformer keeps rows equal
    atom.add(DummyTransformer(strategy="equal"))
    assert atom.features.tolist() == ["x0", "x1", "x2", "x3"]

    # Transformer drops rows
    atom.add(DummyTransformer(strategy="drop"))
    assert atom.features.tolist() == ["x0", "x2", "x3"]

    # Transformer adds rows
    atom.add(DummyTransformer(strategy="add"), columns="!x2")
    assert atom.features.tolist() == ["x0", "x2", "x3", "x4"]


def test_raise_length_mismatch():
    """Assert that an error is raised when there's a mismatch in row length."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(IndexError, match=".*does not match length.*"):
        atom.prune(columns=[2, 4])


def test_add_derivative_columns_keep_position():
    """Assert that derivative columns go after the original."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode(columns="x2")
    assert list(atom.columns[2:5]) == ["x2_a", "x2_b", "x2_d"]


def test_multioutput_y_return():
    """Assert that y returns a dataframe when multioutput."""
    atom = ATOMClassifier(X10, y10_label, stratify=False, random_state=1)
    atom.add(Cleaner())
    assert isinstance(atom.y, pd.DataFrame)

    atom = ATOMClassifier(X10, y10_label, stratify=False, random_state=1)
    atom.add(MultiLabelBinarizer())
    assert isinstance(atom.y, pd.DataFrame)


def test_add_sets_are_kept_equal():
    """Assert that the train and test sets always keep the same rows."""
    atom = ATOMClassifier(X_bin, y_bin, index=True, random_state=1)
    train_idx, test_idx = atom.train.index, atom.test.index
    atom.add(Pruner())
    assert all(idx in train_idx for idx in atom.train.index)
    pd.testing.assert_index_equal(test_idx, atom.test.index)


def test_add_reset_index():
    """Assert that the indices are reset when index=False."""
    atom = ATOMClassifier(X_bin, y_bin, index=False, random_state=1)
    atom.prune()
    assert list(atom.dataset.index) == list(range(len(atom.dataset)))


def test_add_params_to_method():
    """Assert that atom's parameters are passed to the method."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.scale()
    assert atom.pipeline[0].verbose == 1


def test_add_pipeline():
    """Assert that adding a pipeline adds every individual step."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("sfm", SelectFromModel(RandomForestClassifier())),
        ]
    )
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(pipeline)
    assert isinstance(atom.pipeline[0], StandardScaler)
    assert isinstance(atom.pipeline[1], SelectFromModel)


def test_apply():
    """Assert that a function can be applied to the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    atom.apply(np.exp, columns=0)
    assert atom.iat[0, 0] == np.exp(X_bin.iat[0, 0])


# Test data cleaning transformers =================================== >>

def test_balance_wrong_task():
    """Assert that an error is raised for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        atom.balance()


def test_balance_multioutput_task():
    """Assert that an error is raised for multioutput tasks."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    with pytest.raises(ValueError, match=".*not support multioutput.*"):
        atom.balance()


def test_balance():
    """Assert that the balance method balances the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    length = (atom.y_train == 1).sum()
    atom.balance(strategy="NearMiss")
    assert (atom.y_train == 1).sum() != length
    assert hasattr(atom, "nearmiss")


def test_clean():
    """Assert that the clean method cleans the dataset."""
    atom = ATOMClassifier(X10, y10_sn, stratify=False, random_state=1)
    atom.clean()
    assert len(atom.dataset) == 9
    assert atom.mapping == {"target": {"n": 0, "y": 1}}


def test_discretize():
    """Assert that the discretize method bins the numerical columns."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.discretize()
    assert all(dtype.name == "object" for dtype in atom.X.dtypes)


def test_encode():
    """Assert that the encode method encodes all categorical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    assert all(atom.X[col].dtype.kind in "ifu" for col in atom.X.columns)


def test_impute():
    """Assert that the impute method imputes all missing values."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.impute()
    assert atom.dataset.isna().sum().sum() == 0


def test_normalize():
    """Assert that the normalize method transforms the features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    X = atom.X
    atom.normalize()
    assert not atom.X.equals(X)
    assert hasattr(atom, "yeojohnson")


def test_prune():
    """Assert that the prune method handles outliers in the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    len_train, len_test = len(atom.train), len(atom.test)
    atom.prune(strategy="lof")
    assert len(atom.train) != len_train and len(atom.test) == len_test
    assert hasattr(atom, "lof")


def test_scale():
    """Assert that the scale method normalizes the features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert check_scaling(atom.dataset)
    assert hasattr(atom, "standard")


# Test nlp transformers ============================================ >>

def test_textclean():
    """Assert that the textclean method cleans the corpus."""
    atom = ATOMClassifier(X_text, y10, shuffle=False, random_state=1)
    atom.textclean()
    assert atom["corpus"][0] == "i am in new york"
    assert hasattr(atom, "drops")


def test_textnormalize():
    """Assert that the textnormalize method normalizes the corpus."""
    atom = ATOMClassifier(X_text, y10, shuffle=False, random_state=1)
    atom.textnormalize(stopwords=False, custom_stopwords=["yes"], lemmatize=False)
    assert atom["corpus"][0] == ["I", "àm", "in", "ne'w", "york"]


def test_tokenize():
    """Assert that the tokenize method tokenizes the corpus."""
    atom = ATOMClassifier(X_text, y10, shuffle=False, random_state=1)
    atom.tokenize()
    assert atom["corpus"][0] == ["I", "àm", "in", "ne", "'", "w", "york"]


def test_vectorize():
    """Assert that the vectorize method converts the corpus to numerical."""
    atom = ATOMClassifier(X_text, y10, test_size=0.25, random_state=1)
    atom.vectorize(strategy="hashing", n_features=5)
    assert "corpus" not in atom
    assert atom.shape == (10, 6)
    assert hasattr(atom, "hashing")


# Test feature engineering transformers ============================ >>

def test_feature_extraction():
    """Assert that the feature_extraction method creates datetime features."""
    atom = ATOMClassifier(X10_dt, y10, random_state=1)
    atom.feature_extraction(fmt="%d/%m/%Y")
    assert atom.X.shape[1] == 6


def test_feature_generation():
    """Assert that the feature_generation method creates extra features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_generation(n_features=2)
    assert atom.X.shape[1] == X_bin.shape[1] + 2


def test_feature_grouping():
    """Assert that the feature_grouping method group features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_grouping(group=[[0, 1], [1, 2]])
    assert atom.X.shape[1] == X_bin.shape[1] - 3 + 12
    assert hasattr(atom, "groups")


def test_feature_generation_attributes():
    """Assert that the attrs from feature_generation are passed to atom."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_generation("gfg", n_features=2, population_size=30, hall_of_fame=10)
    assert hasattr(atom, "gfg")
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
    assert atom.pipeline[0].univariate.score_func.__name__ == "f_classif"

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy="univariate", solver=None, n_features=8)
    assert atom.pipeline[0].univariate.score_func.__name__ == "f_regression"


def test_default_solver_from_task():
    """Assert that the solver is inferred from the task when a model is selected."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="rfe", solver="tree", n_features=8)
    assert atom.pipeline[0].rfe.estimator_.__class__.__name__ == "DecisionTreeClassifier"

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy="rfe", solver="tree", n_features=25)
    assert atom.pipeline[0].rfe.estimator_.__class__.__name__ == "DecisionTreeRegressor"


@patch("atom.feature_engineering.SequentialFeatureSelector")
def test_default_scoring(cls):
    """Assert that the scoring is atom's metric when exists."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="recall")
    atom.branch = "fs_branch"
    atom.feature_selection(strategy="sfs", solver="lgb", n_features=25)
    assert atom.pipeline[0].kwargs["scoring"].name == "recall"


# Test training methods ============================================ >>

def test_non_numerical_target_column():
    """Assert that an error is raised when the target column is categorical."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    with pytest.raises(ValueError, match=".*target column is not numerical.*"):
        atom.run("Tree")


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
    """Assert that the scaling is passed to atom."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.scale("minmax")
    atom.run("LGB")
    pd.testing.assert_frame_equal(atom.dataset, atom.lgb.dataset)


def test_models_are_replaced():
    """Assert that models with the same name are replaced."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["OLS", "Tree"])
    atom.run("OLS")
    assert atom.models == ["Tree", "OLS"]


def test_models_and_metric_are_updated():
    """Assert that the models and metric attributes are updated correctly."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["OLS", "Tree"], metric=get_scorer("max_error"))
    assert atom.models == ["OLS", "Tree"]
    assert atom.metric == "max_error"

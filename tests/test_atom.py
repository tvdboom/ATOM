"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for atom.py

"""

import glob
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from category_encoders.target_encoder import TargetEncoder
from pandas.testing import assert_frame_equal, assert_index_equal
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, MultiLabelBinarizer, OneHotEncoder, StandardScaler,
)
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.summarize import WindowSummarizer

from atom import ATOMClassifier, ATOMForecaster, ATOMRegressor
from atom.data_cleaning import Cleaner, Pruner
from atom.training import DirectClassifier
from atom.utils.utils import check_scaling

from .conftest import (
    X10, DummyTransformer, X10_dt, X10_nan, X10_str, X10_str2, X20_out, X_bin,
    X_class, X_ex, X_label, X_pa, X_reg, X_sparse, X_text, y10, y10_label,
    y10_label2, y10_sn, y10_str, y_bin, y_class, y_ex, y_fc, y_label,
    y_multiclass, y_multireg, y_reg,
)


# Test __init__ ==================================================== >>

def test_task_assignment():
    """Assert that the correct task is assigned."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.task.name == "binary_classification"

    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.task.name == "multiclass_classification"

    atom = ATOMClassifier(X_label, y=y_label, random_state=1)
    assert atom.task.name == "multilabel_classification"

    atom = ATOMClassifier(X10, y=y10_label, random_state=1)
    assert atom.task.name == "multilabel_classification"

    atom = ATOMClassifier(X10, y=y10_label2, random_state=1)
    assert atom.task.name == "multilabel_classification"

    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    assert atom.task.name == "multiclass_multioutput_classification"

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    assert atom.task.name == "regression"

    atom = ATOMRegressor(X_class, y=y_multiclass, random_state=1)
    assert atom.task.name == "multioutput_regression"


def test_raise_one_target_value():
    """Assert that error raises when there is only one target value."""
    with pytest.raises(ValueError, match=".*1 target value.*"):
        ATOMClassifier(X_bin, [1] * len(X_bin), random_state=1)


def test_backend_with_n_jobs_1():
    """Assert that a warning is raised."""
    with pytest.warns(UserWarning, match=".*Leaving n_jobs=1 ignores.*"):
        ATOMClassifier(X_bin, y_bin, warnings=True, backend="threading", random_state=1)


# Test magic methods =============================================== >>

def test_init():
    """Assert that the __init__ method works for non-standard parameters."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=2, device="gpu", backend="multiprocessing")
    assert atom.device == "gpu"
    assert atom.backend == "multiprocessing"


def test_repr():
    """Assert that the __repr__ method visualizes the pipeline(s)."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert "Branches: main" in str(atom)
    atom.branch = "b2"
    assert "Branches:\n   --> main\n   --> b2 !" in str(atom)


def test_iter():
    """Assert that we can iterate over atom's pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    assert list(atom) == list(atom.pipeline.named_steps.values())


# Test utility properties =========================================== >>

def test_branch():
    """Assert that we can get the current branch."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    assert atom.branch.name == "main"


def test_branch_same():
    """Assert that we can stay on the same branch."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    atom.branch = "main"
    assert atom.branch.name == "main"


def test_branch_change():
    """Assert that we can change to another branch."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    atom.branch = "b2"
    atom.clean()
    atom.branch = "main"
    assert atom.pipeline.steps == []  # Has no Cleaner


def test_branch_existing_name():
    """Assert that an error is raised when the name already exists."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    atom.branch = "b2"
    with pytest.raises(ValueError, match=".*already exists.*"):
        atom.branch = "b2_from_main"


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
    atom.branch = "b3_from_main"
    assert atom.branch.name == "b3"
    assert atom.n_nans > 0


def test_missing():
    """Assert that the missing property returns the values considered 'missing'."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert "NA" in atom.missing


def test_missing_setter():
    """Assert that we can change the missing property."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.missing = (1, 2)
    assert isinstance(atom.missing, list)
    assert "NA" not in atom.missing


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
    """Assert that the nans property returns a series of missing values."""
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


def test_unavailable_sparse_properties():
    """Assert that certain properties are unavailable for sparse datasets."""
    atom = ATOMClassifier(X_sparse, y10, random_state=1)
    with pytest.raises(AttributeError):
        print(atom.nans)
    with pytest.raises(AttributeError):
        print(atom.n_nans)
    with pytest.raises(AttributeError):
        print(atom.outliers)
    with pytest.raises(AttributeError):
        print(atom.n_outliers)


def test_unavailable_regression_properties():
    """Assert that certain properties are unavailable for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(AttributeError):
        print(atom.classes)
    with pytest.raises(AttributeError):
        print(atom.n_classes)


# Test utility methods ============================================= >>

def test_checks():
    """Assert that the checks method works as expected."""
    atom = ATOMForecaster(y_fc, random_state=1)
    checks = atom.checks()
    assert isinstance(checks, pd.DataFrame)


@pytest.mark.parametrize("distributions", [None, "norm", ["norm", "pearson3"]])
def test_distribution(distributions):
    """Assert that the distribution method works as expected."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    dist = atom.distributions(distributions=distributions, columns=(0, 1))
    assert isinstance(dist, pd.DataFrame)


@patch("sweetviz.analyze")
def test_eda_analyze(cls):
    """Assert that the eda method creates a report for one dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.eda(rows="test", filename="report")
    cls.assert_called_once()


@patch("sweetviz.compare")
def test_eda_compare(cls):
    """Assert that the eda method creates a report for two datasets."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.eda(rows={"train": "train", "test": "test"})
    cls.assert_called_once()


def test_eda_invalid_rows():
    """Assert that an error is raised with more than two datasets."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*maximum number of.*"):
        atom.eda(rows=("train", "test", "train"))


def test_inverse_transform():
    """Assert that the inverse_transform method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    atom.scale()
    atom.impute()  # Does nothing, but doesn't crash either
    assert_frame_equal(atom.inverse_transform(atom.X), X_bin)


def test_inverse_transform_output():
    """Assert that the output type is determined by the data engine."""
    atom = ATOMClassifier(X_bin, y_bin, engine="pyarrow", random_state=1)
    atom.scale()
    assert isinstance(atom.inverse_transform(X_bin), pa.Table)


def test_load_no_atom():
    """Assert that an error is raised when the instance is not atom."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.save("trainer")
    with pytest.raises(ValueError, match=".*ATOMClassifier, ATOMRegressor nor.*"):
        ATOMClassifier.load("trainer")


def test_load_already_contains_data():
    """Assert that an error is raised when data is provided without needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save("atom", save_data=True)
    with pytest.raises(ValueError, match=".*already contains data.*"):
        ATOMClassifier.load("atom", data=(X_bin, y_bin))


def test_load_transform_data():
    """Assert that the data is transformed correctly."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale(columns=slice(3, 10))
    atom.apply(np.exp, columns=2)
    atom.feature_generation(strategy="dfs", n_features=5)
    atom.feature_selection(strategy="sfm", solver="lgb", n_features=10)
    atom.save("atom", save_data=False)

    atom2 = ATOMClassifier.load("atom", data=(X_bin, y_bin))
    assert atom2.dataset.shape == atom.dataset.shape


def test_load_transform_data_multiple_branches():
    """Assert that the data is transformed with multiple branches."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    atom.prune()
    atom.branch = "b2"
    atom.balance()
    atom.feature_generation(strategy="dfs", n_features=5)
    atom.branch = "b3"
    atom.feature_selection(strategy="sfm", solver="lgb", n_features=20)
    atom.save("atom_2", save_data=False)

    atom2 = ATOMClassifier.load("atom_2", data=(X_bin, y_bin))

    assert_frame_equal(atom2.og.X, X_bin)
    for branch in atom._branches:
        assert_frame_equal(
            left=atom2._branches[branch.name]._data.data,
            right=atom._branches[branch.name]._data.data,
            check_dtype=False,
        )


def test_reset():
    """Assert that the reset method deletes models and branches."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.scale()
    atom.branch = "2"
    atom.encode()
    atom.run("LR")
    atom.reset(hard=True)
    assert not atom.models
    assert len(atom._branches) == 1
    assert atom["x2"].dtype.name == "object"  # Is reset back to str


def test_save_data():
    """Assert that the dataset is saved to a csv file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save_data("auto", rows="test")
    atom.save_data("auto", rows=range(100))
    assert glob.glob("ATOMClassifier_test.csv")
    assert glob.glob("ATOMClassifier.csv")


def test_shrink_dtypes_excluded():
    """Assert that some dtypes are excluded from changing."""
    X = X_bin.copy()
    X["date"] = pd.date_range(start="1/1/2018", periods=len(X))

    atom = ATOMClassifier(X, y_bin, random_state=1)
    assert atom.dtypes[-2].name == "datetime64[ns]"
    atom.shrink()
    assert atom.dtypes[-2].name == "datetime64[ns]"  # Unchanged


def test_shrink_str2cat():
    """Assert that the str2cat parameter works as intended."""
    atom = ATOMClassifier(X10_str2, y10, random_state=1)
    atom.shrink(str2cat=False)
    assert atom.dtypes[2].name == "string"

    atom.shrink(str2cat=True)
    assert atom.dtypes[2].name == "category"


def test_shrink_int2bool():
    """Assert that the int2bool parameter works as intended."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.dtypes[0].name == "int64"

    atom.shrink(int2bool=True)
    assert atom.dtypes[0].name == "boolean"


def test_shrink_int2uint():
    """Assert that the int2uint parameter works as intended."""
    atom = ATOMClassifier(X10_str2, y10, random_state=1)
    assert atom.dtypes[0].name == "int64"

    atom.shrink(int2uint=False)
    assert atom.dtypes[0].name == "Int8"

    atom.shrink(int2uint=True)
    assert atom.dtypes[0].name == "UInt8"


def test_shrink_sparse_arrays():
    """Assert that sparse arrays are also transformed."""
    atom = ATOMClassifier(X_sparse, y10, random_state=1)
    assert atom.dtypes[0].name == "Sparse[int64, 0]"
    atom.shrink()
    assert atom.dtypes[0].name == "Sparse[int8, 0]"


def test_shrink_dtypes_unchanged():
    """Assert that optimal dtypes are left unchanged."""
    atom = ATOMClassifier(X_bin.astype("Float32"), y_bin, random_state=1)
    assert atom.dtypes[3].name == "Float32"
    atom.shrink()
    assert atom.dtypes[3].name == "Float32"


def test_shrink_dense2sparse():
    """Assert that the dataset can be converted to sparse."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.dtypes[0].name == "float64"
    atom.shrink(dense2sparse=True)
    assert atom.dtypes[0].name.startswith("Sparse[float32")


def test_shrink_pyarrow():
    """Assert that it works with pyarrow dtypes."""
    atom = ATOMClassifier(X_pa, y_bin, engine="pandas-pyarrow", random_state=1)
    assert atom.dtypes[0].name == "double[pyarrow]"
    atom.shrink()
    assert atom.dtypes[0].name == "float[pyarrow]"


def test_shrink_exclude_columns():
    """Assert that columns can be excluded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.dtypes[0].name == "float64"
    assert atom.dtypes[-1].name != "Int8"
    atom.shrink(columns=-1)
    assert atom.dtypes[0].name == "float64"
    assert atom.dtypes[-1].name == "Int8"


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


def test_transform_output():
    """Assert that the output type is determined by the data engine."""
    atom = ATOMClassifier(X_bin, y_bin, engine="pyarrow", random_state=1)
    atom.scale()
    assert isinstance(atom.transform(X_bin), pa.Table)


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


def test_add_basetransformer_params_are_attached():
    """Assert that the n_jobs and random_state params from atom are used."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(PCA)  # When class
    atom.add(PCA(random_state=2))  # When instance
    assert atom.pipeline[0].get_params()["random_state"] == 1
    assert atom.pipeline[1].get_params()["random_state"] == 2


def test_add_results_from_cache():
    """Assert that cached transformers are retrieved."""
    atom = ATOMClassifier(X_bin, y_bin, memory=True, random_state=1)
    atom.scale()

    atom = ATOMClassifier(X_bin, y_bin, memory=True, random_state=1)
    atom.scale()


def test_add_train_only():
    """Assert that atom accepts transformers for the train set only."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(StandardScaler(), train_only=True)
    assert check_scaling(atom.X_train)
    assert not check_scaling(atom.X_test)

    len_train, len_test = len(atom.train), len(atom.test)
    atom.add(Pruner(), train_only=True)
    assert len(atom.train) != len_train
    assert len(atom.test) == len_test


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


def test_only_y_transformation():
    """Assert that only the target column can be transformed."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.scale(columns=-1)
    assert check_scaling(atom.y)


def test_only_y_transformation_return_series():
    """Assert that the output is correctly converted to a dataframe."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.add(WindowSummarizer(), columns=-1)
    assert isinstance(atom.y, pd.Series)
    assert isinstance(atom.dataset, pd.DataFrame)


def test_only_y_transformation_multioutput():
    """Assert that only the target columns can be transformed for multioutput."""
    atom = ATOMRegressor(X_reg, y=y_multireg, random_state=1)
    atom.scale(columns=[-3, -1])
    assert check_scaling(atom.y.iloc[:, [0, 2]])
    assert list(atom.y.columns) == ["a", "b", "c"]


def test_X_and_y_transformation():
    """Assert that only the features are transformed when y is also provided."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.scale(columns=[-2, -1])
    assert check_scaling(atom.X.iloc[:, -1])
    assert not check_scaling(atom.y)


def test_returned_column_already_exists():
    """Assert that an error is raised if an existing column is returned."""

    def func_test(df):
        df["mean texture"] = 1
        return df

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*already exists in the original.*"):
        atom.apply(func_test, columns="!mean texture")


def test_ignore_columns():
    """Assert that columns can be ignored from transformations."""
    atom = ATOMRegressor(X_reg, y_reg, ignore="age", random_state=1)
    atom.scale()
    atom.run("OLS")
    assert "age" in atom
    assert "age" not in atom.pipeline.named_steps["scaler"].feature_names_in_
    assert "age" not in atom.ols.estimator.feature_names_in_


def test_add_sparse_matrices():
    """Assert that transformers that return sp.matrix are accepted."""
    ohe = OneHotEncoder(handle_unknown="ignore").set_output(transform="default")
    atom = ATOMClassifier(X10_str, y10, shuffle=False, random_state=1)
    atom.add(ohe, columns=2)
    assert atom.shape == (10, 8)  # Creates 4 extra columns


def test_add_keep_column_names():
    """Assert that the column names are kept after transforming."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)

    # Transformer has method get_feature_names_out
    atom.add(TargetEncoder(return_df=False))
    assert atom.features.tolist() == ["x0", "x1", "x2", "x3"]

    # Transformer keeps rows equal
    atom.add(DummyTransformer(strategy="equal"), feature_names_out=None)
    assert atom.features.tolist() == ["x0", "x1", "x2", "x3"]

    # Transformer drops rows
    atom.add(DummyTransformer(strategy="drop"), feature_names_out=None)
    assert atom.features.tolist() == ["x0", "x2", "x3"]

    # Transformer adds a new column
    atom.add(DummyTransformer(strategy="add"), columns="!x2", feature_names_out=None)
    assert atom.features.tolist() == ["x0", "x2", "x3", "x4"]


def test_raise_length_mismatch():
    """Assert that an error is raised when there's a mismatch in row length."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(IndexError, match=".*does not match length.*"):
        atom.prune(columns=[2, 4])


def test_keep_pyarrow_dtypes():
    """Assert that columns keep the pyarrow dtype."""
    atom = ATOMClassifier(X_pa, y_bin, random_state=1)
    assert isinstance(atom.dtypes[0], pd.ArrowDtype)
    atom.scale()
    assert isinstance(atom.dtypes[0], pd.ArrowDtype)


def test_add_derivative_columns_keep_position():
    """Assert that derivative columns go after the original."""
    atom = ATOMClassifier(X10_str, y10, shuffle=False, random_state=1)
    atom.encode(columns="x2")
    assert list(atom.columns[2:5]) == ["x2_b", "x2_a", "x2_c"]


def test_multioutput_y_return():
    """Assert that y returns a dataframe when multioutput."""
    atom = ATOMClassifier(X10, y10_label, random_state=1)
    atom.add(Cleaner())
    assert isinstance(atom.y, pd.DataFrame)

    atom = ATOMClassifier(X10, y10_label, random_state=1)
    atom.add(MultiLabelBinarizer())
    assert isinstance(atom.y, pd.DataFrame)


def test_add_sets_are_kept_equal():
    """Assert that the train and test sets always keep the same rows."""
    atom = ATOMClassifier(X_bin, y_bin, index=True, random_state=1)
    train_idx, test_idx = atom.train.index, atom.test.index
    atom.add(Pruner())
    assert all(idx in train_idx for idx in atom.train.index)
    assert_index_equal(test_idx, atom.test.index)


def test_add_reset_index():
    """Assert that the indices are reset when index=False."""
    atom = ATOMClassifier(X_bin, y_bin, index=False, random_state=1)
    atom.prune()
    assert list(atom.dataset.index) == list(range(len(atom.dataset)))


def test_add_raise_duplicate_indices():
    """Assert that an error is raised when indices are duplicated."""

    class AddRowsTransformer(BaseEstimator):
        def transform(self, X, y):
            return pd.concat([X, X.iloc[:5]]), pd.concat([y, y.iloc[:5]])

    atom = ATOMClassifier(X_bin, y_bin, index=True, random_state=1)
    with pytest.raises(ValueError, match=".*Duplicate indices.*"):
        atom.add(AddRowsTransformer)


def test_add_params_to_method():
    """Assert that atom's parameters are passed to the method."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.scale()
    assert atom.pipeline[0].verbose == 1


def test_add_wrap_fit():
    """Assert that sklearn attributes are added to the estimator."""
    atom = ATOMForecaster(X_ex, y=y_ex, random_state=1)
    atom.add(Imputer())
    assert hasattr(atom.pipeline[0], "feature_names_in_")
    assert hasattr(atom.pipeline[0], "n_features_in_")

    # Check there's no double wrapping
    atom.add(Imputer())
    assert hasattr(atom.pipeline[1].fit, "__wrapped__")
    assert not hasattr(atom.pipeline[1].fit.__wrapped__, "__wrapped__")


def test_add_wrap_get_feature_names_out_one_to_one():
    """Assert that get_feature_names_out is added to the estimator."""
    atom = ATOMForecaster(X_ex, y=y_ex, random_state=1)
    atom.add(Imputer(), feature_names_out="one-to-one")
    assert hasattr(atom.pipeline[0], "get_feature_names_out")
    assert list(atom.pipeline[0].get_feature_names_out()) == list(X_ex.columns)


def test_add_wrap_get_feature_names_out_callable():
    """Assert that get_feature_names_out is added to the estimator."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.add(Imputer(), columns=-1, feature_names_out=lambda _: ["test"])
    assert hasattr(atom.pipeline[0], "get_feature_names_out")
    assert list(atom.pipeline[0].get_feature_names_out()) == ["test"]


def test_add_pipeline():
    """Assert that adding a pipeline adds every individual step."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("sfm", SelectFromModel(RandomForestClassifier())),
        ],
    )
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(pipeline)
    assert isinstance(atom.pipeline[0], StandardScaler)
    assert isinstance(atom.pipeline[1], SelectFromModel)


def test_attributes_are_attached():
    """Assert that the transformer's attributes are attached to the branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    atom.scale()
    assert hasattr(atom, "standard_")
    atom.branch = "main"
    assert not hasattr(atom, "standard_")


def test_apply():
    """Assert that a function can be applied to the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    atom.apply(np.exp, columns=0)
    assert atom.iloc[0, 0] == np.exp(X_bin.iloc[0, 0])


# Test data cleaning transformers =================================== >>

def test_balance_wrong_task():
    """Assert that an error is raised for regression and multioutput tasks."""
    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        atom.balance()

    # For multioutput tasks
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        atom.balance()


def test_balance():
    """Assert that the balance method balances the training set."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    atom.clean()  # To have column mapping
    atom.balance(strategy="NearMiss")
    assert (atom.y_train == 0).sum() == (atom.y_train == 1).sum()


def test_clean():
    """Assert that the clean method cleans the dataset."""
    atom = ATOMClassifier(X10, y10_sn, stratify=False, random_state=1)
    atom.clean()
    assert len(atom.dataset) == 9
    assert atom.mapping == {"target": {"n": 0, "y": 1}}


def test_decompose():
    """Assert that the decompose method works."""
    atom = ATOMForecaster(y_fc, sp=12, random_state=1)
    atom.decompose(columns=-1)
    assert atom.dataset.iloc[0, 0] != atom.og.dataset.iloc[0, 0]


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


def test_prune():
    """Assert that the prune method handles outliers in the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    len_train, len_test = len(atom.train), len(atom.test)
    atom.prune(strategy="lof")
    assert len(atom.train) != len_train
    assert len(atom.test) == len_test


def test_scale():
    """Assert that the scale method normalizes the features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert check_scaling(atom.dataset)


# Test nlp transformers ============================================ >>

def test_textclean():
    """Assert that the textclean method cleans the corpus."""
    atom = ATOMClassifier(X_text, y10, shuffle=False, random_state=1)
    atom.textclean()
    assert atom["corpus"][0] == "i am in new york"


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
    atom.feature_grouping({"g1": [0, 1], "g2": "mean.*"})
    assert atom.n_features == 32


def test_default_solver_univariate():
    """Assert that the default solver is selected for strategy="univariate"."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="univariate", solver=None, n_features=8)
    assert atom.pipeline[0].univariate_.score_func.__name__ == "f_classif"

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy="univariate", solver=None, n_features=8)
    assert atom.pipeline[0].univariate_.score_func.__name__ == "f_regression"


def test_default_solver_from_task():
    """Assert that the solver is inferred from the task when a model is selected."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="rfe", solver="tree", n_features=8)
    assert atom.pipeline[0].rfe_.estimator_.__class__.__name__ == "DecisionTreeClassifier"

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy="rfe", solver="tree", n_features=25)
    assert atom.pipeline[0].rfe_.estimator_.__class__.__name__ == "DecisionTreeRegressor"


@patch("atom.feature_engineering.SequentialFeatureSelector", MagicMock())
def test_default_scoring():
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
    assert_frame_equal(atom.dataset, atom.lgb.dataset)


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

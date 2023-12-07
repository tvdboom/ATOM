"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for baserunner.py

"""

import glob
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import (
    assert_frame_equal, assert_index_equal, assert_series_equal,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from atom import ATOMClassifier, ATOMForecaster, ATOMRegressor
from atom.branch import Branch
from atom.training import DirectClassifier, DirectForecaster
from atom.utils.utils import NotFittedError, merge

from .conftest import (
    X10, X_bin, X_class, X_idx, X_label, X_reg, bin_test, bin_train, fc_test,
    fc_train, y10, y_bin, y_class, y_fc, y_idx, y_label, y_multiclass, y_reg,
)


# Test magic methods =============================================== >>

def test_getstate_and_setstate():
    """Assert that versions are checked and a warning raised."""
    atom = ATOMClassifier(X_bin, y_bin, warnings=True)
    atom.run("LR")
    atom.save("atom")

    sys.modules["sklearn"].__version__ = "1.2.7"  # Fake version
    with pytest.warns(Warning, match=".*while the version in this environment.*"):
        ATOMClassifier.load("atom")


def test_getattr_branch():
    """Assert that branches can be called."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    assert atom.b2 is atom._branches["b2"]


def test_getattr_attr_from_branch():
    """Assert that branch attributes can be called."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.pipeline is atom.branch.pipeline


def test_getattr_model():
    """Assert that the models can be called as attributes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert atom.tree is atom._models[0]


def test_getattr_column():
    """Assert that the columns can be accessed as attributes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert isinstance(atom.alcohol, pd.Series)


def test_getattr_dataframe():
    """Assert that the dataset attributes can be called."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(atom.head(), pd.DataFrame)


def test_getattr_invalid():
    """Assert that an error is raised when there is no such attribute."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(AttributeError, match=".*object has no attribute.*"):
        _ = atom.invalid


def test_setattr_to_branch():
    """Assert that branch properties can be set."""
    new_dataset = merge(X_bin, y_bin)
    new_dataset.iloc[0, 3] = 4  # Change one value

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.dataset = new_dataset
    assert atom.dataset.iloc[0, 3] == 4  # Check the value is changed


def test_setattr_normal():
    """Assert that attributes can be set normally."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.attr = "test"
    assert atom.attr == "test"


def test_delattr_models():
    """Assert that models can be deleted through del."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    del atom.lr
    assert atom.models == "MNB"


def test_delattr_normal():
    """Assert that attributes can be deleted normally."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    del atom._config
    assert not hasattr(atom, "_config")


def test_contains():
    """Assert that we can test if a trainer contains a column."""
    trainer = DirectClassifier(models="LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert "mean radius" in trainer


def test_len():
    """Assert that the length of a trainer is the length of the dataset."""
    trainer = DirectClassifier(models="LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert len(trainer) == len(X_bin)


def test_getitem_no_dataset():
    """Assert that an error is raised when getitem is used before run."""
    trainer = DirectClassifier(models="LR", random_state=1)
    with pytest.raises(RuntimeError, match=".*has no dataset.*"):
        print(trainer[4])


def test_getitem_int():
    """Assert that getitem works for a column index."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert_series_equal(atom[0], atom["mean radius"])


def test_getitem_str_from_branch():
    """Assert that getitem works for a branch name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom["main"] is atom._branches["main"]


def test_getitem_str_from_model():
    """Assert that getitem works for a model name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LDA")
    assert atom["lda"] is atom.lda


def test_getitem_str_from_column():
    """Assert that getitem works for a column name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert_series_equal(atom["mean radius"], atom.dataset["mean radius"])


def test_getitem_invalid_str():
    """Assert that an error is raised when getitem is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*has no branch, model or column.*"):
        print(atom["invalid"])


def test_getitem_list():
    """Assert that getitem works for a list of column names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(atom[["mean radius", "mean texture"]], pd.DataFrame)


# Test utility properties ========================================== >>

def test_branch_property():
    """Assert that the branch property returns the current branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(atom.branch, Branch)


def test_delete_last_branch():
    """Assert that an error is raised when the last branch is deleted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(PermissionError, match=".*last branch.*"):
        del atom.branch


def test_delete_depending_models():
    """Assert that dependent models are deleted with the branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    atom.run("LR")
    del atom.branch
    assert not atom.models


def test_delete_current():
    """Assert that we can delete the current branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    del atom.branch
    assert "b2" not in atom._branches


def test_models_property():
    """Assert that the models property returns the model names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree"])
    assert atom.models == ["LR", "Tree"]


def test_metric_property():
    """Assert that the metric property returns the metric names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="f1")
    assert atom.metric == "f1"


def test_winners_property():
    """Assert that the winners property returns the best models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree", "LDA"])
    assert atom.winners == [atom.lr, atom.lda, atom.tree]


def test_winner_property():
    """Assert that the winner property returns the best model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree", "LDA"])
    assert atom.winner is atom.lr


def test_winner_deleter():
    """Assert that the winning model can be deleted through del."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree", "LDA"])
    del atom.winner
    assert atom.models == ["Tree", "LDA"]


def test_results_property():
    """Assert that the results property returns an overview of the results."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert atom.results.shape == (1, 4)


def test_results_property_dropna():
    """Assert that the results property doesn't return columns with NaNs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert "mean_bootstrap" not in atom.results


def test_results_property_successive_halving():
    """Assert that the results property works for successive halving runs."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.successive_halving(["OLS", "Tree"])
    assert atom.results.shape == (3, 4)
    assert list(atom.results.index.get_level_values(0)) == [0.5, 0.5, 1.0]


def test_results_property_train_sizing():
    """Assert that the results property works for train sizing runs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train_sizing("LR")
    assert atom.results.shape == (5, 4)
    assert list(atom.results.index.get_level_values(0)) == [0.2, 0.4, 0.6, 0.8, 1.0]


# Test _set_index ================================================== >>

def test_index_is_true():
    """Assert that the indices are left as is when index=True."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, shuffle=False, random_state=1)
    assert atom.dataset.index[0] == "index_0"


def test_index_is_False():
    """Assert that the indices are reset when index=False."""
    atom = ATOMClassifier(X_idx, y_idx, index=False, shuffle=False, random_state=1)
    assert atom.dataset.index[0] == 0


def test_index_is_int_invalid():
    """Assert that an error is raised when the index is an invalid int."""
    with pytest.raises(IndexError, match=".*is out of range.*"):
        ATOMClassifier(X_bin, y_bin, index=1000, random_state=1)


def test_index_is_int():
    """Assert that a column can be selected from a position."""
    X = X_bin.copy()
    X["mean radius"] = range(len(X))
    atom = ATOMClassifier(X, y_bin, index=0, random_state=1)
    assert atom.dataset.index.name == "mean radius"


def test_index_is_str_invalid():
    """Assert that an error is raised when the index is an invalid str."""
    with pytest.raises(ValueError, match=".*not found in the dataset.*"):
        ATOMClassifier(X_bin, y_bin, index="invalid", random_state=1)


def test_index_is_str():
    """Assert that a column can be selected from a name."""
    X = X_bin.copy()
    X["mean texture"] = range(len(X))
    atom = ATOMClassifier(X, y_bin, index="mean texture", random_state=1)
    assert atom.dataset.index.name == "mean texture"


def test_index_is_range():
    """Assert that a column can be selected from a name."""
    atom = ATOMClassifier(X_bin, y_bin, index=range(len(X_bin)), shuffle=False)
    assert list(atom.dataset.index) == list(range(len(X_bin)))


def test_index_is_target():
    """Assert that an error is raised when the index is the target column."""
    with pytest.raises(ValueError, match=".*same as the target column.*"):
        ATOMRegressor(X_bin, index="worst fractal dimension", random_state=1)


def test_index_is_sequence_no_data_sets_invalid_length():
    """Assert that an error is raised when len(index) != len(data)."""
    with pytest.raises(IndexError, match=".*Length of index.*"):
        ATOMClassifier(X_bin, y_bin, index=[1, 2, 3], random_state=1)


def test_index_is_sequence_no_data_sets():
    """Assert that a sequence is set as index when provided."""
    index = [f"index_{i}" for i in range(len(X_bin))]
    atom = ATOMClassifier(X_bin, y_bin, index=index, random_state=1)
    assert atom.dataset.index[0] == "index_242"


def test_index_is_sequence_has_data_sets_invalid_length():
    """Assert that an error is raised when len(index) != len(data)."""
    with pytest.raises(IndexError, match=".*Length of index.*"):
        ATOMClassifier(bin_train, bin_test, index=[1, 2, 3], random_state=1)


def test_index_is_sequence_has_data_sets():
    """Assert that a sequence is set as index when provided."""
    index = [f"index_{i}" for i in range(len(bin_train) + 2 * len(bin_test))]
    atom = ATOMClassifier(bin_train, bin_test, bin_test, index=index, random_state=1)
    assert atom.dataset.index[0] == "index_0"
    assert atom.holdout.index[0] == "index_569"


def test_duplicate_indices():
    """Assert that an error is raised when there are duplicate indices."""
    with pytest.raises(ValueError, match=".*duplicate indices.*"):
        ATOMClassifier(X_bin, X_bin, index=True, random_state=1)


# Test _get_stratify_columns======================================== >>

@pytest.mark.parametrize("stratify", [True, -1, "target", [-1]])
def test_stratify_options(stratify):
    """Assert that the data can be stratified among data sets."""
    atom = ATOMClassifier(X_bin, y_bin, stratify=stratify, random_state=1)
    train_balance = atom.classes["train"][0] / atom.classes["train"][1]
    test_balance = atom.classes["test"][0] / atom.classes["test"][1]
    np.testing.assert_almost_equal(train_balance, test_balance, decimal=2)


def test_stratify_is_False():
    """Assert that the data is not stratified when stratify=False."""
    atom = ATOMClassifier(X_bin, y_bin, stratify=False, random_state=1)
    train_balance = atom.classes["train"][0] / atom.classes["train"][1]
    test_balance = atom.classes["test"][0] / atom.classes["test"][1]
    assert abs(train_balance - test_balance) > 0.05


def test_stratify_invalid_column_int():
    """Assert that an error is raised when the value is invalid."""
    with pytest.raises(ValueError, match=".*out of range for a dataset.*"):
        ATOMClassifier(X_bin, y_bin, stratify=100, random_state=1)


def test_stratify_invalid_column_str():
    """Assert that an error is raised when the value is invalid."""
    with pytest.raises(ValueError, match=".*not found in the dataset.*"):
        ATOMClassifier(X_bin, y_bin, stratify="invalid", random_state=1)


# Test _get_data =================================================== >>

def test_input_is_y_without_arrays():
    """Assert that input y through parameter works."""
    atom = ATOMForecaster(y=y_fc, random_state=1)
    assert atom.dataset.shape == (len(y_fc), 1)


def test_empty_data_arrays():
    """Assert that an error is raised when no data is provided."""
    with pytest.raises(ValueError, match=".*data arrays are empty.*"):
        ATOMClassifier(n_rows=100, random_state=1)


def test_data_already_set():
    """Assert that if there already is data, the call to run can be empty."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    trainer.run()
    assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))
    assert_index_equal(trainer.branch._data.train_idx, bin_train.index)
    assert_index_equal(trainer.branch._data.test_idx, bin_test.index)


def test_input_is_X():
    """Assert that input X works."""
    atom = ATOMRegressor(X_bin, random_state=1)
    assert atom.dataset.shape == X_bin.shape


def test_input_is_y():
    """Assert that input y works for forecasting tasks."""
    atom = ATOMForecaster(y_fc, random_state=1)
    assert atom.dataset.shape == (len(y_fc), 1)


def test_input_is_X_with_parameter_y():
    """Assert that input X can be combined with parameter y."""
    atom = ATOMRegressor(X_bin, y="mean texture", random_state=1)
    assert atom.target == "mean texture"


def test_input_invalid_holdout():
    """Assert that an error is raised when holdout is invalid."""
    with pytest.raises(ValueError, match=".*holdout_size parameter.*"):
        ATOMClassifier(X_bin, test_size=0.3, holdout_size=0.8)


@pytest.mark.parametrize("holdout_size", [0.1, 40])
def test_input_is_X_with_holdout(holdout_size):
    """Assert that input X can be combined with a holdout set."""
    atom = ATOMRegressor(X_bin, holdout_size=holdout_size, random_state=1)
    assert isinstance(atom.holdout, pd.DataFrame)


@pytest.mark.parametrize("shuffle", [True, False])
def test_input_is_train_test_with_holdout(shuffle):
    """Assert that input train and test can be combined with a holdout set."""
    atom = ATOMClassifier(bin_train, bin_test, bin_test, shuffle=shuffle)
    assert isinstance(atom.holdout, pd.DataFrame)


@pytest.mark.parametrize("n_rows", [0.7, 1])
def test_n_rows_X_y_frac(n_rows):
    """Assert that n_rows<=1 work for input X and X, y."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=n_rows, random_state=1)
    assert len(atom.dataset) == int(len(X_bin) * n_rows)


def test_n_rows_X_y_int():
    """Assert that n_rows>1 work for input X and X, y."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=200, random_state=1)
    assert len(atom.dataset) == 200


def test_n_rows_forecasting():
    """Assert that the rows are cut from the dataset's head when forecasting."""
    atom = ATOMForecaster(y_fc, n_rows=142, random_state=1)
    assert len(atom.dataset) == 142
    assert atom.dataset.index[0] == y_fc.index[len(y_fc) - 142]


def test_n_rows_too_large():
    """Assert that an error is raised when n_rows>len(data)."""
    with pytest.raises(ValueError, match=".*n_rows parameter.*"):
        ATOMClassifier(X_bin, y_bin, n_rows=1e6, random_state=1)


def test_no_shuffle_X_y():
    """Assert that the order is kept when shuffle=False."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False)
    assert_frame_equal(atom.X, X_bin)


def test_length_dataset():
    """Assert that the dataset is always len>=5."""
    with pytest.raises(ValueError, match=".*n_rows=1 for small.*"):
        ATOMClassifier(X10, y10, n_rows=0.01, random_state=1)


@pytest.mark.parametrize("test_size", [-2, 0, 1000])
def test_test_size_parameter(test_size):
    """Assert that the test_size parameter is in correct range."""
    with pytest.raises(ValueError, match=".*test_size parameter.*"):
        ATOMClassifier(X_bin, test_size=test_size, random_state=1)


def test_test_size_fraction():
    """Assert that the test_size parameters splits the sets correctly when <1."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.2, random_state=1)
    assert len(atom.test) == int(0.2 * len(X_bin))
    assert len(atom.train) == len(X_bin) - int(0.2 * len(X_bin))


def test_test_size_int():
    """Assert that the test_size parameters splits the sets correctly when >=1."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=100, random_state=1)
    assert len(atom.test) == 100
    assert len(atom.train) == len(X_bin) - 100


def test_error_message_impossible_stratification():
    """Assert that the correct error is shown when stratification fails."""
    with pytest.raises(ValueError, match=".*stratify=False.*"):
        ATOMClassifier(X_label[:30], y=y_label[:30], stratify=True, random_state=1)


def test_input_is_X_y():
    """Assert that input X, y works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.dataset.shape == merge(X_bin, y_bin).shape


def test_input_is_2_tuples():
    """Assert that the 2 tuples input works."""
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run((X_train, y_train), (X_test, y_test))
    assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))


def test_input_is_train_test():
    """Assert that input train, test works."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))


def test_input_is_train_test_with_parameter_y():
    """Assert that input X works can be combined with y."""
    atom = ATOMClassifier(bin_train, bin_test, y="mean texture", random_state=1)
    assert atom.target == "mean texture"


def test_input_is_train_test_for_forecast():
    """Assert that input train, test works for forecast tasks."""
    trainer = DirectForecaster("ES", errors="raise", random_state=1)
    trainer.run(fc_train, fc_test)
    assert_series_equal(trainer.y, pd.concat([fc_train, fc_test]))


def test_input_is_3_tuples():
    """Assert that the 3 tuples input works."""
    X_train = bin_train.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    X_test = bin_test.iloc[100:-20, :-1]
    y_test = bin_test.iloc[100:-20, -1]
    X_holdout = bin_test.iloc[-20:, :-1]
    y_holdout = bin_test.iloc[-20:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run((X_train, y_train), (X_test, y_test), (X_holdout, y_holdout))
    assert_frame_equal(trainer.X, pd.concat([X_train, X_test]))
    assert_frame_equal(trainer.holdout, pd.concat([X_holdout, y_holdout], axis=1))


def test_input_is_train_test_holdout():
    """Assert that input train, test, holdout works."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test.iloc[:100], bin_test.iloc[100:])
    assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test.iloc[:100]]))
    assert_frame_equal(trainer.holdout, bin_test.iloc[100:])


def test_4_data_provided():
    """Assert that the 4 elements input works."""
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(X_train, X_test, y_train, y_test)
    assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))


def test_6_data_provided():
    """Assert that the 6 elements input works."""
    X_train = bin_train.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    X_test = bin_test.iloc[100:-20, :-1]
    y_test = bin_test.iloc[100:-20, -1]
    X_holdout = bin_test.iloc[-20:, :-1]
    y_holdout = bin_test.iloc[-20:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(X_train, X_test, X_holdout, y_train, y_test, y_holdout)
    assert_frame_equal(trainer.X, pd.concat([X_train, X_test]))
    assert_frame_equal(trainer.holdout, pd.concat([X_holdout, y_holdout], axis=1))


def test_invalid_input():
    """Assert that an error is raised when input arrays are invalid."""
    trainer = DirectClassifier("LR", random_state=1)
    with pytest.raises(ValueError, match=".*Invalid data arrays.*"):
        trainer.run(X_bin, y_bin, X_bin, y_bin, y_bin, X_bin, X_bin)


def test_n_rows_train_test_frac():
    """Assert that n_rows<=1 work for input with train and test."""
    atom = ATOMClassifier(bin_train, bin_test, n_rows=0.8, random_state=1)
    assert len(atom.train) == int(len(bin_train) * 0.8)
    assert len(atom.test) == int(len(bin_test) * 0.8)


def test_no_shuffle_train_test():
    """Assert that the order is kept when shuffle=False."""
    atom = ATOMClassifier(bin_train, bin_test, shuffle=False)
    assert_frame_equal(
        left=atom.train,
        right=bin_train.reset_index(drop=True),
        check_dtype=False,
    )


def test_n_rows_train_test_int():
    """Assert that an error is raised when n_rows>1 for input with train and test."""
    with pytest.raises(ValueError, match=".*must be <1 when the train and test.*"):
        ATOMClassifier(bin_train, bin_test, n_rows=100, random_state=1)


def test_dataset_is_shuffled():
    """Assert that the dataset is shuffled before splitting."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=True, random_state=1)
    assert not X_bin.equals(atom.X)


def test_holdout_is_shuffled():
    """Assert that the holdout set is shuffled."""
    atom = ATOMClassifier(bin_train, bin_test, bin_test, shuffle=True, random_state=1)
    assert not bin_test.equals(atom.holdout)


def test_reset_index():
    """Assert that the indices are reset for the all data sets."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    assert list(atom.dataset.index) == list(range(len(atom.dataset)))


def test_unequal_columns_train_test():
    """Assert that an error is raised when train and test have different columns."""
    with pytest.raises(ValueError, match=".*train and test set do not have.*"):
        ATOMClassifier(X10, bin_test, random_state=1)


def test_unequal_columns_holdout():
    """Assert that an error is raised when holdout has different columns."""
    with pytest.raises(ValueError, match=".*holdout set does not have.*"):
        ATOMClassifier(bin_train, bin_test, X10, random_state=1)


def test_merger_to_dataset():
    """Assert that the merger between X and y was successful."""
    # Reset index since the order of rows is different after shuffling
    merger = X_bin.merge(y_bin.to_frame(), left_index=True, right_index=True)
    df1 = merger.sort_values(by=merger.columns.tolist())

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    df2 = atom.dataset.sort_values(by=atom.dataset.columns.tolist())
    assert_frame_equal(
        left=df1.reset_index(drop=True),
        right=df2.reset_index(drop=True),
        check_dtype=False,
    )


def test_invalid_index_forecast():
    """Assert that an error is raised when the index is invalid."""
    with pytest.raises(ValueError, match=".*index of the dataset must.*"):
        ATOMForecaster(pd.Series([1, 2, 3, 4, 5], index=[1, 4, 2, 3, 5]), random_state=1)


# Test utility methods ============================================= >>

def test_get_models_is_None():
    """Assert that all models are returned by default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR_1", "LR_2"])
    assert atom._get_models(models=None) == [atom.lr_1, atom.lr_2]


def test_get_models_by_int():
    """Assert that models can be selected by index."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(IndexError, match=".*out of range.*"):
        atom._get_models(models=0)
    atom.run(["LR_1", "LR_2"])
    assert atom._get_models(models=1) == [atom.lr_2]


def test_get_models_by_slice():
    """Assert that a slice of models is returned."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train_sizing(["LR_1", "LR_2"])
    assert len(atom._get_models(models=slice(1, 4))) == 3


def test_get_models_winner():
    """Assert that the winner is returned when used as name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    assert atom._get_models(models="winner") == [atom.lr]


def test_get_models_by_str():
    """Assert that models can be retrieved by name or regex."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["GNB", "LR_1", "LR_2"])
    assert atom._get_models("gnb+lr_1") == [atom.gnb, atom.lr_1]
    assert atom._get_models(["gnb+lr_1", "lr_2"]) == [atom.gnb, atom.lr_1, atom.lr_2]
    assert atom._get_models("lr.*") == [atom.lr_1, atom.lr_2]
    assert atom._get_models("!lr_1") == [atom.gnb, atom.lr_2]
    assert atom._get_models("!lr.*") == [atom.gnb]
    with pytest.raises(ValueError, match=".*any model that matches.*"):
        atom._get_models(models="invalid")


def test_get_models_exclude():
    """Assert that models can be excluded using `!`."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*not find any model.*"):
        atom._get_models(models="!invalid")
    atom.run(["LR_1", "LR_2"])
    assert atom._get_models(models="!lr_1") == [atom.lr_2]
    assert atom._get_models(models="!.*_2$") == [atom.lr_1]


def test_get_models_by_model():
    """Assert that a model can be called using a Model instance."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert atom._get_models(models=atom.lr) == [atom.lr]


def test_get_models_include_or_exclude():
    """Assert that an error is raised when models are included and excluded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR_1", "LR_2"])
    with pytest.raises(ValueError, match=".*either include or exclude models.*"):
        atom._get_models(models=["LR_1", "!LR_2"])


def test_get_models_remove_ensembles():
    """Assert that ensembles can be excluded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR_1", "LR_2"])
    atom.voting()
    assert "Vote" not in atom._get_models(models=None, ensembles=False)


def test_get_models_invalid_branch():
    """Assert that an error is raised when the branch is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    atom.branch = "2"
    atom.run("LDA")
    with pytest.raises(ValueError, match=".*have been fitted.*"):
        atom._get_models(models=None, branch=atom.branch)


def test_get_models_remove_duplicates():
    """Assert that duplicate models are returned."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR_1", "LR_2"])
    assert atom._get_models(["LR_1", "LR_1"]) == [atom.lr_1]


def test_available_models():
    """Assert that the available_models method shows the models per task."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    models = atom.available_models()
    assert isinstance(models, pd.DataFrame)
    assert "LR" in models["acronym"].unique()
    assert "BR" not in models["acronym"].unique()  # Is not a classifier


def test_clear():
    """Assert that the clear method resets all model's attributes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    atom.lda.plot_shap_beeswarm(display=False)
    assert not atom.lda._shap._shap_values.empty
    atom.clear()
    assert atom.lda._shap._shap_values.empty


def test_delete_default():
    """Assert that all models in branch are deleted as default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    atom.delete()  # All models
    assert not atom.models
    assert not atom.metric
    assert atom.results.empty


@pytest.mark.parametrize("metric", ["ap", "f1"])
def test_evaluate(metric):
    """Assert that the evaluate method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.evaluate)
    atom.run(["Tree", "SVM"])
    assert isinstance(atom.evaluate(metric), pd.DataFrame)


def test_export_pipeline_same_transformer():
    """Assert that two same transformers get different names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.clean()
    atom.clean()
    pl = atom.export_pipeline()
    assert list(pl.named_steps) == ["cleaner", "cleaner-2", "cleaner-3"]


def test_export_pipeline_with_model():
    """Assert that the model's branch is used."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    atom.run("GNB")
    atom.branch = "b2"
    atom.normalize()
    assert len(atom.export_pipeline(model="GNB")) == 2


def test_get_class_weight_regression():
    """Assert that an error is raised when called from regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(AttributeError):
        atom.get_class_weight()


def test_get_class_weight():
    """Assert that the get_class_weight method returns a dict of the classes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert list(atom.get_class_weight()) == [0, 1, 2]


def test_get_class_weight_multioutput():
    """Assert that the get_class_weight method works for multioutput."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    assert list(atom.get_class_weight()) == ["a", "b", "c"]


def test_get_sample_weights_regression():
    """Assert that an error is raised when called from regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(AttributeError):
        atom.get_sample_weight()


def test_get_sample_weight():
    """Assert that the get_sample_weight method returns a series."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert len(atom.get_sample_weight()) == len(atom.train)


def test_get_sample_weight_multioutput():
    """Assert that the get_sample_weight method works for multioutput."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    assert len(atom.get_sample_weight()) == len(atom.train)


def test_merge_invalid_class():
    """Assert that an error is raised when the class is not a trainer."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(TypeError, match=".*Expecting a.*"):
        atom.merge(ATOMRegressor(X_reg, y_reg, random_state=1))


def test_merge_different_dataset():
    """Assert that an error is raised when the og dataset is different."""
    atom_1 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_2 = ATOMClassifier(X10, y10, random_state=1)
    with pytest.raises(ValueError, match=".*different dataset.*"):
        atom_1.merge(atom_2)


def test_merge_adopts_metrics():
    """Assert that the metric of the merged instance is adopted."""
    atom_1 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_2 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_2.run("Tree", metric="f1")
    atom_1.merge(atom_2)
    assert atom_1.metric == "f1"


def test_merge_different_metrics():
    """Assert that an error is raised when the metrics are different."""
    atom_1 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_1.run("Tree", metric="f1")
    atom_2 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_2.run("Tree", metric="auc")
    with pytest.raises(ValueError, match=".*different metric.*"):
        atom_1.merge(atom_2)


def test_merge():
    """Assert that the merger handles branches, models and attributes."""
    atom_1 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_1.run("Tree")
    atom_2 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_2.branch.name = "b2"
    atom_2.missing = ["missing"]
    atom_2.run("LR")
    atom_1.merge(atom_2)
    assert list(atom_1._branches) == [atom_1.main, atom_1.b2]
    assert atom_1.models == ["Tree", "LR"]
    assert atom_1.missing[-1] == "missing"


def test_merge_with_suffix():
    """Assert that the merger handles branches, models and attributes."""
    atom_1 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_1.run(["Tree", "LDA"], n_trials=1, ht_params={"distributions": {"LDA": "test"}})
    atom_2 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_2.run(["Tree", "LDA"], n_trials=1, ht_params={"distributions": {"LDA": "test"}})
    atom_1.merge(atom_2)
    assert list(atom_1._branches) == [atom_1.main, atom_1.main2]
    assert atom_1.models == ["Tree", "Tree2"]


def test_file_is_saved():
    """Assert that the pickle file is saved."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save("auto")
    assert glob.glob("ATOMClassifier.pkl")


@patch("atom.baserunner.pickle", MagicMock())
def test_save_data_false():
    """Assert that the dataset is restored after saving with save_data=False."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.save(filename="atom", save_data=False)
    assert atom.dataset is not None  # Dataset is restored after saving
    assert atom.holdout is not None  # Holdout is restored after saving


def test_stacking():
    """Assert that the stacking method creates a Stack model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.stacking)
    atom.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    atom.stacking()
    assert hasattr(atom, "stack")
    assert "Stack" in atom.models


def test_stacking_non_ensembles():
    """Assert that stacking ignores other ensembles."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    atom.voting()
    atom.stacking()
    assert len(atom.stack.estimator.estimators) == 2  # No voting


def test_stacking_invalid_models():
    """Assert that an error is raised when <2 models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    with pytest.raises(ValueError, match=".*contain at least two.*"):
        atom.stacking()


def test_stacking_invalid_name():
    """Assert that an error is raised when the model already exists."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree"])
    atom.stacking()
    with pytest.raises(ValueError, match=".*multiple Stacking.*"):
        atom.stacking()


def test_stacking_custom_models():
    """Assert that stacking can be created selecting the models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.stacking)
    atom.run(["LR", "LDA", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    atom.stacking(models=["LDA", "LGB"])
    assert list(atom.stack._models) == [atom.lda, atom.lgb]


def test_stacking_different_name():
    """Assert that the acronym is added in front of the new name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    atom.stacking(name="stack_1")
    atom.stacking(name="_2")
    assert hasattr(atom, "Stack_1")
    assert hasattr(atom, "Stack_2")


def test_stacking_unknown_predefined_final_estimator():
    """Assert that an error is raised when the final estimator is unknown."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    with pytest.raises(ValueError, match=".*Unknown model.*"):
        atom.stacking(final_estimator="invalid")


def test_stacking_invalid_predefined_final_estimator():
    """Assert that an error is raised when the final estimator is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    with pytest.raises(ValueError, match=".*can not perform.*"):
        atom.stacking(final_estimator="OLS")


def test_stacking_predefined_final_estimator():
    """Assert that the final estimator accepts predefined models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    atom.stacking(final_estimator="LDA")
    assert isinstance(atom.stack.estimator.final_estimator_, LDA)


def test_voting():
    """Assert that the voting method creates a Vote model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.voting)
    atom.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    atom.voting(name="2")
    assert hasattr(atom, "Vote2")
    assert "Vote2" in atom.models


def test_voting_invalid_name():
    """Assert that an error is raised when the model already exists."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree"])
    atom.voting()
    with pytest.raises(ValueError, match=".*multiple Voting.*"):
        atom.voting()


def test_voting_invalid_models():
    """Assert that an error is raised when <2 models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    with pytest.raises(ValueError, match=".*contain at least two.*"):
        atom.voting()

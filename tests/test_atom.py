# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.data_cleaning import Scaler, Pruner
from atom.utils import check_scaling
from .utils import (
    FILE_DIR, X_bin, y_bin, X_class, y_class, X_reg, y_reg, X_text,
    y_text, X10, X10_nan, X10_str, X10_str2, X10_dt, y10, y10_str,
    y10_sn, X20_out,
)


# Test __init__ ==================================================== >>

def test_dtypes_shrinkage():
    """Assert that the dtypes are optimized to save memory."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert not atom.dtypes.equals(X_bin.dtypes)


def test_minimal_dtypes_unchanged():
    """Assert that the minimal dtypes are left unchanged."""
    X = X_bin.astype("float32")
    atom = ATOMClassifier(X, y_bin, random_state=1)
    assert atom.X.dtypes.equals(X.dtypes)


def test_dtypes_cat_or_object():
    """Assert that categorical columns change dtype if necessary."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.dtypes[2].name != "category"

    atom = ATOMClassifier(X10_str2, y10, random_state=1)
    assert atom.dtypes[2].name == "category"


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
    assert atom.mapping == {"n": "n", "nan": np.NaN, "y": "y"}


# Test magic methods =============================================== >>

def test_repr():
    """Assert that the __repr__ method visualizes the pipeline(s)."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert "Branches: master" in str(atom)
    atom.branch = "branch_2"
    assert "Branches:\n   >>> master\n   >>> branch_2 !" in str(atom)


def test_iter():
    """Assert that we can iterate over atom's pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    assert [item for item in atom][1] == atom.pipeline[1]


def test_contains():
    """Assert that we can test if atom contains a column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert "mean radius" in atom


def test_len():
    """Assert that the length of atom is the length of the pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    assert len(atom) == 1


def test_getitem():
    """Assert that atom is subscriptable."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    assert atom[1].__class__.__name__ == "Imputer"
    assert atom["mean radius"].equals(atom.dataset["mean radius"])
    with pytest.raises(TypeError, match=r".*subscriptable with types.*"):
        print(atom[2.3])


# Test utility properties =========================================== >>

def test_branch_setter_empty():
    """Assert that an error is raised when the name is empty."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    with pytest.raises(ValueError, match=r".*Can't create a branch.*"):
        atom.branch = ""


def test_branch_model_acronym():
    """Assert that an error is raised when the name is a models' acronym."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    with pytest.raises(ValueError, match=r".*acronym of model.*"):
        atom.branch = "Lda"


def test_branch_restricted_name():
    """Assert that an error is raised when the name is restricted."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    with pytest.raises(ValueError, match=r".*This name is reserved.*"):
        atom.branch = "og"


def test_branch_setter_change():
    """Assert that we can change to an old branch."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    atom.branch = "branch_2"
    atom.clean()
    atom.branch = "master"
    assert atom.pipeline.empty  # Has no clean estimator


def test_branch_setter_new():
    """Assert that we can create a new branch."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    atom.clean()
    atom.branch = "branch_2"
    assert list(atom._branches.keys()) == ["og", "master", "branch_2"]


def test_branch_setter_from_valid():
    """Assert that we can create a new branch, not from the current one."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.branch = "branch_2"
    atom.impute()
    atom.branch = "branch_3_from_master"
    assert atom.branch.name == "branch_3"
    assert atom.n_nans > 0


def test_branch_setter_from_invalid():
    """Assert that an error is raised when the from branch doesn't exist."""
    atom = ATOMClassifier(X10, y10, random_state=1)
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

def test_status():
    """Assert that the status method prints an overview of the instance."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.status()


def test_reset():
    """Assert that the reset method deletes models and branches."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.branch = "2"
    atom.run("LR")
    atom.reset()
    assert not atom.models and len(atom._branches) == 2
    assert atom.dataset.equals(atom.og.dataset)


@pytest.mark.parametrize("column", ["Feature 1", 1])
def test_distribution(column):
    """Assert that the distribution method and file are created."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    pytest.raises(ValueError, atom.distribution, column="Feature 3")
    df = atom.distribution(column=column)
    assert len(df) == 11


@patch("pandas_profiling.ProfileReport")
def test_report(cls):
    """Assert that the report method and file are created."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.report(n_rows=10, filename="report")
    cls.return_value.to_file.assert_called_once_with("report.html")


def test_transform_method():
    """ Assert that the transform method works as intended."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode(max_onehot=None)
    assert atom.transform(X10_str)["Feature 3"].dtype.kind in "ifu"


def test_pipeline_parameter_None():
    """Assert that only some transformers are used."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.impute(strat_num="median")
    atom.prune(max_sigma=2)
    X = atom.transform(X10_nan, pipeline=None)  # Only use imputer
    assert len(X) == 10


def test_pipeline_parameter_True():
    """Assert that all transformers are used."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.impute(strat_num="median")
    atom.prune(max_sigma=2)
    X = atom.transform(X10_nan, pipeline=True)  # Use both transformers
    assert len(X) < 10


def test_pipeline_parameter_False():
    """Assert that no transformers used."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.impute(strat_num="median")
    atom.prune(max_sigma=2)
    X = atom.transform(X10_nan, pipeline=False)  # Use None
    assert isinstance(X, list)  # X is unchanged


def test_verbose_raises_when_invalid():
    """Assert an error is raised for an invalid value of verbose."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    pytest.raises(ValueError, atom.transform, X_bin, verbose=3)


def test_transform_with_y():
    """Assert that the transform method works when y is provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.prune(strategy="iforest", include_target=True)
    X, y = atom.transform(X_bin, y_bin, pipeline=[0])
    assert len(y) < len(y_bin)


@patch("tpot.TPOTClassifier")
def test_automl_classification(cls):
    """Assert that the automl method works for classification tasks."""
    pl = Pipeline(
        steps=[
            ("standardscaler", StandardScaler()),
            ("robustscaler", RobustScaler()),
            ("mlpclassifier", MLPClassifier(alpha=0.001, random_state=1)),
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
        verbosity=0,
    )
    assert len(atom) == 2
    assert atom.models == ["Tree", "MLP"]


@patch("tpot.TPOTRegressor")
def test_automl_regression(cls):
    """Assert that the automl method works for regression tasks."""
    pl = Pipeline(
        steps=[
            ("rbfsampler", RBFSampler(gamma=0.95, random_state=2)),
            ("lassolarscv", LassoLarsCV(normalize=False)),
        ]
    )
    cls.return_value.fitted_pipeline_ = pl.fit(X_reg, y_reg)

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.automl(scoring="r2", random_state=2)
    cls.assert_called_with(n_jobs=1, scoring="r2", verbosity=0, random_state=2)
    assert atom.metric == "r2"


def test_automl_invalid_scoring():
    """Assert that an error is raised when the provided scoring is invalid."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree", metric="mse")
    pytest.raises(ValueError, atom.automl, scoring="r2")


def test_save_data():
    """Assert that the dataset is saved to a csv file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save_data(FILE_DIR + "auto")
    assert glob.glob(FILE_DIR + "ATOMClassifier_dataset.csv")


@pytest.mark.parametrize("pl", [(None, 1), (True, 2), ([0], 1)])
def test_export_pipeline(pl):
    """Assert that we can export the pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.prune()
    assert len(atom.export_pipeline(pipeline=pl[0])) == pl[1]


def test_export_empty_pipeline():
    """Assert that an error is raised when the pipeline is empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    pytest.raises(ValueError, atom.export_pipeline, pipeline=False)


def test_export_pipeline_verbose():
    """Assert that the verbosity is passed to the transformers."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.run("LGB")
    assert atom.export_pipeline("LGB", verbose=2)[0].verbose == 2


def test_export_same_transformer():
    """Assert that two same transformers get different names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.clean()
    atom.clean()
    pl = atom.export_pipeline()
    assert list(pl.named_steps.keys()) == ["cleaner", "cleaner2", "cleaner3"]


def test_export_pipeline_scaler():
    """Assert that a scaler is included in the pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["GNB", "LGB"])
    assert not isinstance(atom.export_pipeline("GNB")[0], Scaler)
    assert isinstance(atom.export_pipeline("LGB")[0], Scaler)


# Test base transformers =========================================== >>

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


def test_drop_target_column():
    """Assert that an error is raised when the target column is dropped."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.drop, columns=-1)


def test_drop():
    """Assert that columns can be dropped through the pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.drop([0, 1])
    assert atom.n_features == X_bin.shape[1] - 2
    assert str(atom.pipeline[0]).startswith("DropTransformer(columns")


def test_apply_not_callable():
    """Assert that an error is raised when func is not callable."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(TypeError, atom.apply, func=RandomForestClassifier(), columns=0)


def test_apply_same_column():
    """Assert that apply can transform an existing column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.apply(lambda x: 1, columns=0)
    assert atom["mean radius"].sum() == atom.shape[0]
    assert str(atom.pipeline[0]).startswith("FuncTransformer(func=<lambda>")


def test_apply_new_column():
    """Assert that apply can create a new column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.apply(lambda x: 1, columns="new column")
    assert atom["new column"].sum() == atom.shape[0]


def test_apply_args_and_kwargs():
    """Assert that args and kwargs are passed to the function."""

    def test_func(df, arg_1, arg_2="mean radius"):
        return df[arg_2] + arg_1

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.apply(test_func, columns="new column", args=(10,), arg_2="mean texture")
    assert atom["new column"][0] == atom["mean texture"][0] + 10


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


def test_no_transformer():
    """Assert that an error is raised if the estimator has no transformer."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.add, RandomForestClassifier())


def test_basetransformer_params_are_attached():
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


def test_transformer_only_y():
    """Assert that atom accepts transformers with only an y parameter."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    atom.add(LabelEncoder())
    assert np.all((atom["Target"] == 0) | (atom["Target"] == 1))


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


def test_derivative_columns_keep_position():
    """Assert that derivative columns go after the original."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    assert atom.columns[2].startswith("Feature 3")


def test_duplicate_columns_are_ignored():
    """Assert that duplicate columns are ignored for the transformers."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.add(StandardScaler(), columns=["mean radius", "mean radius"])
    assert not atom["mean radius"].equals(X_bin["mean radius"])


def test_sets_are_kept_equal():
    """Assert that the train and test sets always keep the same rows."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    len_train, len_test = len(atom.train), len(atom.test)
    atom.add(Pruner())
    assert len(atom.train) < len_train and len(atom.test) < len_test


# Test data cleaning transformers =================================== >>

def test_scale():
    """Assert that the scale method normalizes the features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert check_scaling(atom.dataset)
    assert hasattr(atom, "standard")


def test_gauss():
    """Assert that the gauss method transforms the features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    X = atom.X.copy()
    atom.gauss()
    assert not atom.X.equals(X)
    assert hasattr(atom, "yeojohnson")


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
    atom.prune(strategy="lof")
    assert len(atom.train) != len_train and len(atom.test) == len_test
    assert hasattr(atom, "lof")


def test_balance_wrong_task():
    """Assert that an error is raised for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(PermissionError, atom.balance, oversample=0.7)


def test_balance():
    """Assert that the balance method balances the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    length = (atom.y_train == 1).sum()
    atom.balance(strategy="NearMiss")
    assert (atom.y_train == 1).sum() != length
    assert atom.pipeline[0].mapping == atom.mapping
    assert hasattr(atom, "nearmiss")


# Test nlp transformers ============================================ >>

def test_textclean():
    """Assert that the textclean method cleans the corpus."""
    atom = ATOMClassifier(X_text, y_text, random_state=1)
    atom.textclean()
    assert atom["Corpus"][0] == "yes sir "
    assert hasattr(atom, "drops")


def test_tokenize():
    """Assert that the tokenize method tokenizes the corpus."""
    atom = ATOMClassifier(X_text, y_text, random_state=1)
    atom.tokenize()
    assert atom["Corpus"][0] == ["yes", "sir", "12"]


def test_normalize():
    """Assert that the normalize method normalizes the corpus."""
    atom = ATOMClassifier(X_text, y_text, random_state=1)
    atom.normalize(stopwords=False, custom_stopwords=["yes"])
    assert atom["Corpus"][0] == ["sir", "12"]


def test_vectorize():
    """Assert that the vectorize method vectorizes the corpus."""
    atom = ATOMClassifier(X_text, y_text, random_state=1)
    atom.vectorize(strategy="hashing", n_features=5)
    assert "Corpus" not in atom
    assert atom.shape == (4, 6)
    assert hasattr(atom, "hashing")


# Test feature engineering transformers ============================ >>

def test_feature_extraction():
    """Assert that the feature_extraction method creates datetime features."""
    atom = ATOMClassifier(X10_dt, y10, random_state=1)
    atom.feature_extraction()
    assert atom.X.shape[1] == 6


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

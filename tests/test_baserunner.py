# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for baserunner.py

"""

import sys
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

from atom import ATOMClassifier, ATOMRegressor
from atom.branch import Branch
from atom.training import DirectClassifier
from atom.utils import NotFittedError, merge

from .conftest import (
    X10, X_bin, X_class, X_label, X_reg, bin_test, bin_train, y10, y_bin,
    y_class, y_label, y_multiclass, y_multireg, y_reg,
)


# Test magic methods =============================================== >>

def test_getstate_and_setstate():
    """Assert that versions are checked and a warning raised."""
    atom = ATOMClassifier(X_bin, y_bin, warnings=True)
    atom.run("LR")
    atom.save("atom")

    sys.modules["sklearn"].__version__ = "1.0.0"  # Fake version
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
    new_dataset.iat[0, 3] = 4  # Change one value

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.dataset = new_dataset
    assert atom.dataset.iat[0, 3] == 4  # Check the value is changed


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
    del atom.index
    assert not hasattr(atom, "index")


def test_contains():
    """Assert that we can test if a trainer contains a column."""
    trainer = DirectClassifier(models="LR", random_state=1)
    assert "mean radius" not in trainer
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
    assert atom[0] is atom["mean radius"]


def test_getitem_str_from_branch():
    """Assert that getitem works for a branch name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom["master"] is atom._branches["master"]


def test_getitem_str_from_model():
    """Assert that getitem works for a model name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LDA")
    assert atom["lda"] is atom.lda


def test_getitem_str_from_column():
    """Assert that getitem works for a column name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom["mean radius"] is atom.dataset["mean radius"]


def test_getitem_invalid_str():
    """Assert that an error is raised when getitem is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*has no branch, model or column.*"):
        print(atom["invalid"])


def test_getitem_list():
    """Assert that getitem works for a list of column names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(atom[["mean radius", "mean texture"]], pd.DataFrame)


def test_getitem_invalid_type():
    """Assert that an error is raised when getitem is invalid type."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(TypeError, match=".*subscriptable with types.*"):
        print(atom[2.3])


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


def test_sample_weight():
    """Assert that we can get the sample weight from the branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.sample_weight is None


def test_sample_weight_setter_dict():
    """Assert that we can change the sample weight."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.sample_weight is None


def test_multioutput_None():
    """Assert that the multioutput estimator is ignored when None."""
    atom = ATOMClassifier(X_label, y=y_label, random_state=1)
    atom.multioutput = None
    atom.run("MLP")  # MLP has native support for multilabel
    assert atom.mlp.estimator.__class__.__name__ == "MLPClassifier"


@pytest.mark.parametrize("multioutput", [MultiOutputRegressor, RegressorChain(None)])
def test_multioutput_regression(multioutput):
    """Assert that the multioutput estimator works for regression tasks."""
    atom = ATOMRegressor(X_reg, y=y_multireg, random_state=1)
    atom.multioutput = multioutput
    atom.run("OLS")
    assert atom.models == "OLS"


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
    atom.run(["LR", "Tree", "LGB"])
    assert atom.winners == [atom.lr, atom.tree, atom.lgb]


def test_winner_property():
    """Assert that the winner property returns the best model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree", "LGB"])
    assert atom.winner is atom.lr


def test_winner_deleter():
    """Assert that the winning model can be deleted through del."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree", "LGB"])
    del atom.winner
    assert atom.models == ["Tree", "LGB"]


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
    """Assert that the results works for successive halving runs."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.successive_halving(["OLS", "Tree"])
    assert atom.results.shape == (3, 4)
    assert list(atom.results.index.get_level_values(0)) == [0.5, 0.5, 1.0]


def test_results_property_train_sizing():
    """Assert that the results works for train sizing runs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train_sizing("LR")
    assert atom.results.shape == (5, 4)
    assert list(atom.results.index.get_level_values(0)) == [0.2, 0.4, 0.6, 0.8, 1.0]


# Test utility methods ============================================= >>

def test_get_models_is_None():
    """Assert that all models are returned by default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom._get_models(models=None) == []
    atom.run(["LR1", "LR2"])
    assert atom._get_models(models=None) == [atom.lr1, atom.lr2]


def test_get_models_by_int():
    """Assert that models can be selected by index."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*out of range for a pipeline.*"):
        atom._get_models(models=0)
    atom.run(["LR1", "LR2"])
    assert atom._get_models(models=1) == [atom.lr2]


def test_get_models_by_slice():
    """Assert that a slice of models is returned."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train_sizing(["LR1", "LR2"])
    assert len(atom._get_models(models=slice(1, 4))) == 3


def test_get_models_winner():
    """Assert that the winner is returned when used as name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR1", "LR2"])
    assert atom._get_models(models="winner") == [atom.lr1]


def test_get_models_by_str():
    """Assert that models can be retrieved by name or regex."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["GNB", "LR1", "LR2"])
    assert atom._get_models(models="gnb+lr1") == [atom.gnb, atom.lr1]
    assert atom._get_models(models=["gnb+lr1", "lr2"]) == [atom.gnb, atom.lr1, atom.lr2]
    assert atom._get_models(models="lr.*") == [atom.lr1, atom.lr2]
    assert atom._get_models(models="!lr1") == [atom.gnb, atom.lr2]
    assert atom._get_models(models="!lr.*") == [atom.gnb]
    with pytest.raises(ValueError, match=".*any model that matches.*"):
        atom._get_models(models="invalid")


def test_get_models_invalid_type():
    """Assert that an error is raised when the type is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(TypeError, match=".*Invalid type for the models.*"):
        atom._get_models(models=[3.2])


def test_get_models_exclude():
    """Assert that models can be excluded using `!`."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*not find any model.*"):
        atom._get_models(models="!invalid")
    atom.run(["LR1", "LR2"])
    assert atom._get_models(models="!lr1") == [atom.lr2]
    assert atom._get_models(models="!.*2$") == [atom.lr1]


def test_get_models_by_model():
    """Assert that a model can be called using a Model instance."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert atom._get_models(models=atom.lr) == [atom.lr]


def test_get_models_wrong_type():
    """Assert that an error is raised when the wrong type is used."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    with pytest.raises(TypeError, match=".*type for the models parameter.*"):
        atom._get_models(models=atom)


def test_get_models_include_or_exclude():
    """Assert that an error is raised when models are included and excluded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR1", "LR2"])
    with pytest.raises(ValueError, match=".*either include or exclude models.*"):
        atom._get_models(models=["LR1", "!LR2"])


def test_get_models_remove_ensembles():
    """Assert that ensembles can be excluded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR1", "LR2"])
    atom.voting()
    assert "Vote" not in atom._get_models(models=None, ensembles=False)


def test_get_models_invalid_branch():
    """Assert that an error is raised when the branch is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    atom.branch = "2"
    atom.run("LDA")
    with pytest.raises(ValueError, match=".*must have been fitted.*"):
        atom._get_models(models=None, branch=atom.branch)


def test_get_models_remove_duplicates():
    """Assert that duplicate models are returned."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR1", "LR2"])
    assert atom._get_models(["LR1", "LR1"]) == [atom.lr1]


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
    atom.run(["LR", "LGB"])
    atom.lgb.plot_shap_beeswarm(display=False)
    assert "predict_proba_train" in atom.lr.__dict__
    assert not atom.lgb._shap._shap_values.empty
    atom.clear()
    assert "predict_proba_train" not in atom.lr.__dict__
    assert atom.lgb._shap._shap_values.empty


def test_delete_default():
    """Assert that all models in branch are deleted as default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.delete()  # No models
    atom.run(["LR", "LDA"])
    atom.delete()  # All models
    assert not (atom.models or atom.metric)
    assert atom.results.empty


@pytest.mark.parametrize("metric", ["ap", "roc_auc_ovo", "f1"])
def test_evaluate(metric):
    """Assert that the evaluate method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.evaluate)
    atom.run(["Tree", "SVM"])
    assert isinstance(atom.evaluate(metric), pd.DataFrame)


def test_export_pipeline_empty():
    """Assert that an error is raised when the pipeline is empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(RuntimeError, match=".*no pipeline to export.*"):
        atom.export_pipeline()


def test_export_pipeline_verbose():
    """Assert that the verbosity is passed to the transformers."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    assert atom.export_pipeline(verbose=2)[0].verbose == 2


def test_export_pipeline_same_transformer():
    """Assert that two same transformers get different names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.clean()
    atom.clean()
    pl = atom.export_pipeline()
    assert list(pl.named_steps) == ["cleaner", "cleaner2", "cleaner3"]


def test_export_pipeline_with_model():
    """Assert that the model's branch is used."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    atom.run("GNB")
    atom.branch = "b2"
    atom.normalize()
    assert len(atom.export_pipeline(model="GNB")) == 2


@patch("tempfile.gettempdir")
def test_export_pipeline_memory(func):
    """Assert that memory is True triggers a temp dir."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    atom.export_pipeline(memory=True)
    func.assert_called_once()


def test_get_class_weights_regression():
    """Assert that an error is raised when called from regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(AttributeError):
        atom.get_class_weight()


def test_class_weights_invalid_dataset():
    """Assert that an error is raised if invalid value for dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*the dataset parameter.*"):
        atom.get_class_weight("invalid")


@pytest.mark.parametrize("dataset", ["train", "test", "dataset"])
def test_get_class_weights(dataset):
    """Assert that the get_class_weight method returns a dict of the classes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert list(atom.get_class_weight(dataset)) == [0, 1, 2]


def test_get_class_weight_multioutput():
    """Assert that the get_class_weight method works for multioutput."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    assert list(atom.get_class_weight()) == ["a", "b", "c"]


def test_merge_invalid_class():
    """Assert that an error is raised when the class is not a trainer."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(TypeError, match=".*Expecting a.*"):
        atom.merge(LDA())


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
    assert list(atom_1._branches) == [atom_1.master, atom_1.b2]
    assert atom_1.models == ["Tree", "LR"]
    assert atom_1.missing[-1] == "missing"


def test_merge_with_suffix():
    """Assert that the merger handles branches, models and attributes."""
    atom_1 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_1.run(["Tree", "LDA"], n_trials=1, ht_params={"distributions": {"LDA": "test"}})
    atom_2 = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom_2.run(["Tree", "LDA"], n_trials=1, ht_params={"distributions": {"LDA": "test"}})
    atom_1.merge(atom_2)
    assert list(atom_1._branches) == [atom_1.master, atom_1.master2]
    assert atom_1.models == ["Tree", "Tree2"]


def test_stacking():
    """Assert that the stacking method creates a Stack model."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    pytest.raises(NotFittedError, atom.stacking)
    atom.run(["LR", "LGB"])
    atom.stacking()
    assert hasattr(atom, "stack")
    assert "Stack" in atom.models
    assert atom.stack._run


def test_stacking_non_ensembles():
    """Assert that stacking ignores other ensembles."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
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
    atom.run(["LR", "LDA", "LGB"])
    atom.stacking(models=["LDA", "LGB"])
    assert list(atom.stack._models) == [atom.lda, atom.lgb]


def test_stacking_different_name():
    """Assert that the acronym is added in front of the new name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    atom.stacking(name="stack_1")
    atom.stacking(name="_2")
    assert hasattr(atom, "Stack_1") and hasattr(atom, "Stack_2")


def test_stacking_unknown_predefined_final_estimator():
    """Assert that an error is raised when the final estimator is unknown."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    with pytest.raises(ValueError, match=".*Unknown model.*"):
        atom.stacking(final_estimator="invalid")


def test_stacking_invalid_predefined_final_estimator():
    """Assert that an error is raised when the final estimator is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    with pytest.raises(ValueError, match=".*can not perform.*"):
        atom.stacking(final_estimator="OLS")


def test_stacking_predefined_final_estimator():
    """Assert that the final estimator accepts predefined models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    atom.stacking(final_estimator="LDA")
    assert isinstance(atom.stack.estimator.final_estimator_, LDA)


def test_voting():
    """Assert that the voting method creates a Vote model."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    pytest.raises(NotFittedError, atom.voting)
    atom.run(["LR", "LGB"])
    atom.voting(name="2")
    assert hasattr(atom, "Vote2")
    assert "Vote2" in atom.models
    assert atom.vote2._run


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

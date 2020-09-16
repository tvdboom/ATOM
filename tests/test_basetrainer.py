# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for basetrainer.py

"""

# Import packages
import pytest
import pandas as pd
from skopt.space.space import Integer
from sklearn.metrics import get_scorer, f1_score

# Own modules
from atom.training import TrainerClassifier, TrainerRegressor
from .utils import bin_train, bin_test, class_train, class_test, reg_train, reg_test


# Test _check_parameters ==================================================== >>

def test_model_is_string():
    """Assert that a string input is accepted."""
    trainer = TrainerClassifier('LR')
    trainer.run(bin_train, bin_test)
    assert trainer.models == ['LR']


def test_models_get_right_name():
    """Assert that the model names are transformed to the correct acronyms."""
    trainer = TrainerClassifier(['lR', 'et', 'CATB'])
    trainer.run(bin_train, bin_test)
    assert trainer.models == ['LR', 'ET', 'CatB']


def test_duplicate_models():
    """Assert that an error is raised when models contains duplicates."""
    trainer = TrainerClassifier(['lr', 'LR', 'lgb'])
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_only_task_models():
    """Assert that an error is raised for models at invalid task."""
    trainer = TrainerClassifier('OLS')  # Only regression
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)

    trainer = TrainerRegressor('LDA')  # Only classification
    pytest.raises(ValueError, trainer.run, reg_train, reg_test)


def test_n_calls_parameter_invalid():
    """Assert that an error is raised when n_calls is invalid."""
    trainer = TrainerClassifier('LR', n_calls=(2, 2))
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_n_calls_parameter_to_list():
    """Assert that n_calls is made a list."""
    trainer = TrainerClassifier(['LR', 'LDA'], n_calls=7)
    trainer.run(bin_train, bin_test)
    assert trainer.n_calls == [7, 7]


def test_n_initial_points_parameter_invalid():
    """Assert that an error is raised when n_initial_points is invalid."""
    trainer = TrainerClassifier('LR', n_calls=2, n_initial_points=(2, 2))
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_n_initial_points_parameter_to_list():
    """Assert that n_initial_points is made a list."""
    trainer = TrainerClassifier(['LR', 'LDA'], n_calls=2, n_initial_points=1)
    trainer.run(bin_train, bin_test)
    assert trainer.n_initial_points == [1, 1]


def test_bagging_parameter_invalid():
    """Assert that an error is raised when bagging is invalid."""
    trainer = TrainerClassifier('LR', bagging=(2, 2))
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_bagging_parameter_to_list():
    """Assert that bagging is made a list."""
    trainer = TrainerClassifier(['LR', 'LDA'], bagging=2)
    trainer.run(bin_train, bin_test)
    assert trainer.bagging == [2, 2]


def test_dimensions_is_array():
    """Assert that the dimensions parameter works as array."""
    dim = [Integer(100, 1000, name='max_iter')]

    # If more than one model
    trainer = TrainerClassifier(['LR', 'LDA'], bo_params={'dimensions': dim})
    pytest.raises(TypeError, trainer.run, bin_train, bin_test)

    # For a single model
    trainer = TrainerClassifier('LR', bo_params={'dimensions': {'LR': dim}})
    assert trainer.bo_params == {'dimensions': {'LR': dim}}


def test_dimensions_proper_naming():
    """Assert that the correct model acronyms are used as keys."""
    dim = [Integer(100, 1000, name='max_iter')]
    trainer = TrainerClassifier('LR', bo_params={'dimensions': {'lr': dim}})
    trainer.run(bin_train, bin_test)
    assert trainer.bo_params == {'dimensions': {'LR': dim}}


def test_default_metric():
    """Assert that a default metric_ is assigned depending on the task."""
    trainer = TrainerClassifier('LR')
    trainer.run(bin_train, bin_test)
    assert trainer.metric == 'f1'

    trainer = TrainerClassifier('LR')
    trainer.run(class_train, class_test)
    assert trainer.metric == 'f1_weighted'

    trainer = TrainerRegressor('LGB')
    trainer.run(reg_train, reg_test)
    assert trainer.metric == 'r2'


# Test _prepare_metric ====================================================== >>

def test_metric_to_list():
    """Assert that the metric_ attribute is always a list."""
    trainer = TrainerClassifier('LR', metric='f1')
    trainer.run(bin_train, bin_test)
    assert isinstance(trainer.metric_, list)


def test_greater_is_better_parameter():
    """Assert that an error is raised if invalid length for greater_is_better."""
    trainer = TrainerClassifier('LR', greater_is_better=[True, False])
    trainer.run(bin_train, bin_test)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_needs_proba_parameter():
    """Assert that an error is raised if invalid length for needs_proba."""
    trainer = TrainerClassifier('LR', needs_proba=[True, False])
    trainer.run(bin_train, bin_test)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_needs_threshold_parameter():
    """Assert that an error is raised if invalid length for needs_threshold."""
    trainer = TrainerClassifier('LR', needs_threshold=[True, False])
    trainer.run(bin_train, bin_test)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_metric_acronym():
    """"Assert that using the metric_ acronyms work."""
    trainer = TrainerClassifier('LR', metric='auc')
    trainer.run(bin_train, bin_test)
    assert trainer.metric == 'roc_auc'


def test_invalid_scorer_name():
    """Assert that an error is raised when scorer name is invalid."""
    trainer = TrainerClassifier('LR', metric='invalid')
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_function_metric_parameter():
    """Assert that a function metric_ works."""
    trainer = TrainerClassifier('LR', metric=f1_score)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == 'f1_score'


def test_scorer_metric_parameter():
    """Assert that a scorer metric_ works."""
    trainer = TrainerClassifier('LR', metric=get_scorer(f1_score))
    trainer.run(bin_train, bin_test)
    assert trainer.metric == 'f1_score'


# Test _params_to_attr ====================================================== >>

def test_data_already_set():
    """Assert that if there already is data, the call to run can be empty."""
    dataset = pd.concat([bin_train, bin_test]).reset_index(drop=True)

    trainer = TrainerClassifier('LR')
    trainer.run(bin_train, bin_test)
    trainer.run()
    assert trainer._data.equals(dataset)
    assert trainer._idx == [len(bin_train), len(bin_test)]


def test_train_test_provided():
    """Assert that it runs when train and test are provided."""
    dataset = pd.concat([bin_train, bin_test]).reset_index(drop=True)

    trainer = TrainerClassifier('LR')
    trainer.run(bin_train, bin_test)
    assert trainer._data.equals(dataset)


def test_4_data_provided():
    """Assert that it runs when X_train, X_test, etc... are provided."""
    dataset = pd.concat([bin_train, bin_test]).reset_index(drop=True)
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]

    trainer = TrainerClassifier('LR')
    trainer.run(X_train, X_test, y_train, y_test)
    assert trainer._data.equals(dataset)


def test_invalid_input():
    """Assert that an error is raised for an invalid input."""
    trainer = TrainerClassifier('LR')
    pytest.raises(ValueError, trainer.run, bin_train)


# Test _run ================================================================= >>

def test_sequence_parameters():
    """Assert that every model get his corresponding parameters."""
    trainer = TrainerClassifier(['LR', 'LDA', 'LGB'],
                                n_calls=(2, 3, 4),
                                n_initial_points=(1, 2, 3),
                                bagging=[2, 5, 7],
                                random_state=1)
    trainer.run(bin_train, bin_test)
    assert len(trainer.LR.bo) == 2
    assert sum(trainer.LDA.bo.index.str.startswith('Random')) == 2
    assert len(trainer.lgb.metric_bagging) == 7


def test_invalid_n_calls_parameter():
    """Assert that an error is raised for negative n_calls."""
    trainer = TrainerClassifier('LR', n_calls=-1, n_initial_points=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_creation_model_subclasses():
    """Assert that the model subclasses are created correctly."""
    trainer = TrainerClassifier(['LR', 'LDA'])
    trainer.run(bin_train, bin_test)
    assert hasattr(trainer, 'LR') and hasattr(trainer, 'lr')
    assert hasattr(trainer, 'LDA') and hasattr(trainer, 'lda')


def test_scaler_is_created():
    """Assert the scaler is created for models that need scaling."""
    trainer = TrainerClassifier('LGB')
    trainer.run(bin_train, bin_test)
    assert trainer.scaler is not None


def test_error_handling():
    """Assert that models with errors are removed from pipeline."""
    trainer = TrainerClassifier(['LR', 'LDA'], n_calls=4, n_initial_points=[2, -1])
    trainer.run(bin_train, bin_test)
    assert trainer.errors.get('LDA')
    assert 'lDA' not in trainer.models
    assert 'LDA' not in trainer.results.index


def test_all_models_failed():
    """Assert that an error is raised when all models failed."""
    trainer = TrainerClassifier('LR', n_calls=4, n_initial_points=-1)
    pytest.raises(RuntimeError, trainer.run, bin_train, bin_test)

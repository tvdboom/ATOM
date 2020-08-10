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
from .utils import bin_train, bin_test


# Test __init__ ============================================================= >>

def test_model_is_string():
    """Assert that a string input is accepted."""
    trainer = TrainerClassifier('LR')
    assert trainer.models == ['LR']


def test_models_get_right_name():
    """Assert that the model names are transformed to the correct acronyms."""
    trainer = TrainerClassifier(['lr', 'et', 'CATB'])
    assert trainer.models == ['LR', 'ET', 'CatB']


def test_duplicate_models():
    """Assert that an error is raised when models contains duplicates."""
    pytest.raises(ValueError, TrainerClassifier, ['lr', 'LR', 'lgb'])


def test_only_task_models():
    """Assert that an error is raised for models at invalid task."""
    # Only regression
    pytest.raises(ValueError, TrainerClassifier, 'OLS')

    # Only classification
    pytest.raises(ValueError, TrainerRegressor, 'LDA')


def test_n_calls_parameter():
    """Assert that the n_calls parameter is set correctly."""
    pytest.raises(ValueError, TrainerClassifier, 'LR', n_calls=(2, 2))
    trainer = TrainerClassifier(['LR', 'LDA'], n_calls=2)
    assert trainer.n_calls == [2, 2]


def test_n_random_starts_parameter():
    """Assert that the n_random_starts parameter is set correctly."""
    pytest.raises(
        ValueError, TrainerClassifier, 'LR', n_calls=2, n_random_starts=(2, 2))
    trainer = TrainerClassifier(['LR', 'LDA'], n_calls=2, n_random_starts=1)
    assert trainer.n_random_starts == [1, 1]


def test_bagging_parameter():
    """Assert that the bagging parameter is set correctly."""
    pytest.raises(ValueError, TrainerClassifier, 'LR', bagging=(2, 2))
    trainer = TrainerClassifier(['LR', 'LDA'], bagging=2)
    assert trainer.bagging == [2, 2]


def test_dimensions_is_array():
    """Assert that the dimensions parameter works as array."""
    dim = {'dimensions': [Integer(100, 1000, name='max_iter')]}

    # If more than one model
    pytest.raises(TypeError, TrainerClassifier, ['LR', 'LDA'], bo_params=dim.copy())

    # For a single model
    trainer = TrainerClassifier('LR', bo_params=dim.copy())
    assert trainer.bo_params == {'dimensions': {'LR': dim['dimensions']}}


def test_dimensions_proper_naming():
    """Assert that the correct model acronyms are used as keys."""
    lst = [Integer(100, 1000, name='max_iter')]
    trainer = TrainerClassifier('LR', bo_params={'dimensions': {'lr': lst}})
    assert trainer.bo_params == {'dimensions': {'LR': lst}}


# Test _prepare_metric ====================================================== >>

def test_metric_to_list():
    """Assert that the metric_ attribute is always a list."""
    trainer = TrainerClassifier('LR', metric='f1')
    assert isinstance(trainer.metric_, list)


def test_max_3_metrics():
    """Assert that an error is raised when more than 3 metrics."""
    metrics = ['f1', 'recall', 'precision', 'average_precision']
    pytest.raises(ValueError, TrainerClassifier, 'LR', metric=metrics)


def test_greater_is_better_parameter():
    """Assert that an error is raised if invalid length for greater_is_better."""
    pytest.raises(
        ValueError, TrainerClassifier, 'LR', greater_is_better=[True, False])


def test_needs_proba_parameter():
    """Assert that an error is raised if invalid length for needs_proba."""
    pytest.raises(ValueError, TrainerClassifier, 'LR', needs_proba=[True, False])


def test_needs_threshold_parameter():
    """Assert that an error is raised if invalid length for needs_threshold."""
    pytest.raises(ValueError, TrainerClassifier, 'LR', needs_threshold=[True, False])


def test_metric_acronym():
    """"Assert that using the metric_ acronyms work."""
    trainer = TrainerClassifier('LR', metric='auc')
    assert trainer.metric == 'roc_auc'


def test_invalid_scorer_name():
    """Assert that an error is raised when scorer name is invalid."""
    pytest.raises(ValueError, TrainerClassifier, 'LR', metric='invalid')


def test_function_metric_parameter():
    """Assert that a function metric_ works."""
    trainer = TrainerClassifier('LR', metric=f1_score)
    assert trainer.metric == 'f1_score'


def test_scorer_metric_parameter():
    """Assert that a scorer metric_ works."""
    trainer = TrainerClassifier('LR', metric=get_scorer(f1_score))
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
                                n_random_starts=(1, 2, 3),
                                bagging=[2, 5, 7])
    trainer.run(bin_train, bin_test)
    assert len(trainer.LR.bo) == 2
    assert sum(trainer.LDA.bo.index.str.startswith('Random')) == 2
    assert len(trainer.lgb.score_bagging) == 7


def test_invalid_n_calls_parameter():
    """Assert that an error is raised for negative n_calls."""
    trainer = TrainerClassifier('LR', n_calls=-1, n_random_starts=1)
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
    trainer = TrainerClassifier(['LR', 'LDA'], n_calls=4, n_random_starts=[2, -1])
    trainer.run(bin_train, bin_test)
    assert trainer.errors.get('LDA')
    assert 'lDA' not in trainer.models
    assert 'LDA' not in trainer.results.index


def test_all_models_failed():
    """Assert that an error is raised when all models failed."""
    trainer = TrainerClassifier('LR', n_calls=4, n_random_starts=-1)
    pytest.raises(RuntimeError, trainer.run, bin_train, bin_test)

# Test results ============================================================== >>

def test_is_fitted_results():
    """Assert that an error is raised when the class is not fitted."""
    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(AttributeError, atom.results)


def test_error_unknown_metric():
    """Assert that an error is raised when an unknown metric is selected."""
    atom = ATOMRegressor(X_reg, y_reg)
    atom.run(models='lgb', metric='r2')
    pytest.raises(ValueError, atom.results, 'unknown')


def test_error_invalid_metric():
    """Assert that an error is raised when an invalid metric is selected."""
    atom = ATOMRegressor(X_reg, y_reg)
    atom.run(models='lgb', metric='r2')
    pytest.raises(ValueError, atom.results, 'average_precision')


def test_all_tasks():
    """Assert that the method works for all three tasks."""
    # For binary classification
    atom = ATOMClassifier(X_bin, y_bin)
    atom.run(models=['lda', 'lgb'], metric='f1')
    atom.results()
    atom.results('jaccard')
    assert 1 == 1

    # For multiclass classification
    atom = ATOMClassifier(X_class, y_class)
    atom.run(models=['pa', 'lgb'], metric='recall_macro')
    atom.results()
    atom.results('f1_micro')
    assert 2 == 2

    # For regression
    atom = ATOMRegressor(X_reg, y_reg)
    atom.run(models='lgb', metric='neg_mean_absolute_error')
    atom.results()
    atom.results('neg_mean_poisson_deviance')
    assert 3 == 3




def test_invalid_models_parameter():
    """Assert that an error is raised for invalid or duplicate models."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(ValueError, atom.run, models='test')
    pytest.raises(ValueError, atom.run, models=['OLS', 'OLS'])


def test_invalid_task_models_parameter():
    """Assert that an error is raised for models with invalid tasks."""
    # Only classification
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(ValueError, atom.run, models='LDA')

    # Only regression
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.run, models='OLS')


def test_skip_iter_parameter():
    """Assert that an error is raised for negative skip_iter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.successive_halving, 'Tree', skip_iter=-1)


def test_n_calls_invalid_length():
    """Assert that an error is raised when len n_calls != models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.run, 'Tree', n_calls=(3, 2))


def test_n_random_starts_invalid_length():
    """Assert that an error is raised when len n_random_starts != models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.run, 'Tree', n_random_starts=(3, 2))


def test_n_calls_parameter_as_sequence():
    """Assert that n_calls as sequence works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['Tree', 'LGB'], n_calls=(3, 2), n_random_starts=1)
    assert len(atom.Tree.BO) == 3
    assert len(atom.LGB.BO) == 2


def test_n_random_starts_parameter_as_sequence():
    """Assert that n_random_starts as sequence works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['Tree', 'LGB'], n_calls=5, n_random_starts=(3, 1))
    assert (atom.Tree.BO['call'] == 'Random start').sum() == 3
    assert (atom.LGB.BO['call'] == 'Random start').sum() == 1


def test_kwargs_dimensions():
    """Assert that bo_params['dimensions'] raises an error when wrong type."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    kwargs = {'dimensions': 3}
    pytest.raises(TypeError, atom.run, ['Tree', 'LGB'], bo_kwargs=kwargs)


def test_default_metric_parameter():
    """Assert that the correct default metric is set per task."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LR')
    assert atom.metric.name == 'f1'

    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run('LR')
    assert atom.metric.name == 'f1_weighted'

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS')
    assert atom.metric.name == 'r2'


def test_same_metric():
    """Assert that the default metric stays the same if already defined."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS', metric='max_error')
    atom.run('BR')
    assert atom.metric.name == 'max_error'


def test_invalid_metric_parameter():
    """Assert that an error is raised for an unknown metric."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.run, models='LDA', metric='unknown')


def test_function_metric_parameter():
    """Assert that a function metric works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr', metric=f1_score)
    assert not atom.errors


def test_scorer_metric_parameter():
    """Assert that a scorer metric works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('ols', metric=get_scorer('neg_mean_squared_error'))
    assert not atom.errors


def test_invalid_train_sizes():
    """Assert than error is raised when element in train_sizes is >1."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(ValueError, atom.train_sizing, ['OLS'], train_sizes=[0.8, 2])


def test_scores_attribute():
    """Assert that the scores attribute has the right format."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)

    # For a direct pipeline
    atom.run('OLS')
    assert isinstance(atom.results, pd.DataFrame)

    # For successive_halving
    atom.successive_halving(['OLS', 'BR', 'LGB', 'CatB'])
    assert isinstance(atom.results, list)
    assert len(atom.results) == 3

    # For successive_halving
    atom.train_sizing('OLS', train_sizes=[0.3, 0.6])
    assert isinstance(atom.results, list)
    assert len(atom.results) == 2


def test_exception_encountered():
    """Assert that exceptions are attached as attributes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(['BNB', 'LDA'], n_calls=3, n_random_starts=1)
    assert atom.BNB.error
    assert 'BNB' in atom.errors.keys()


def test_exception_removed_models():
    """Assert that models with exceptions are removed from self.models."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(['BNB', 'LDA'], n_calls=3, n_random_starts=1)
    assert 'BNB' not in atom.models


def test_exception_not_subsequent_iterations():
    """Assert that models with exceptions are removed from following iters."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.train_sizing(['LR', 'LGB'], 'f1_macro', n_calls=3, n_random_starts=1)
    assert 'LGB' not in atom.results[-1].index


def test_creation_subclasses_lowercase():
    """Assert that the model subclasses for lowercase are created."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run('LDA')
    assert hasattr(atom, 'lda')
    assert atom.LDA is atom.lda


def test_all_models_failed():
    """Assert than an error is raised when all models encountered errors."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(ValueError, atom.run, 'BNB', n_calls=6)

"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for pipeline.py

"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sktime.proba.normal import Normal

from atom import ATOMClassifier, ATOMForecaster
from atom.pipeline import Pipeline

from .conftest import X_bin, y_bin, y_fc


@pytest.fixture()
def pipeline():
    """Get a pipeline from atom with/without a final estimator."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.encode()
    atom.prune()
    atom.add(StandardScaler(), columns=0)
    atom.run("LR")

    def get_pipeline(model):
        return atom.export_pipeline(model="LR" if model else None)

    return get_pipeline


@pytest.fixture()
def pipeline_ts():
    """Get a forecast pipeline from atom with a final estimator."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.scale(columns=-1)
    atom.run("NF")
    return atom.export_pipeline(model="NF")


def test_getattr(pipeline):
    """Assert that attributes can be fetched from the final estimator."""
    pl = pipeline(model=True)
    assert isinstance(pl.coef_, np.ndarray)

    # Final estimator has no attribute
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        print(pl.test)


def test_fit(pipeline):
    """Assert that the pipeline can be fitted normally."""
    pl = pipeline(model=True)
    assert pl.fit(X_bin, y_bin)
    pl.steps.insert(1, ("passthrough", None))
    assert pl.fit(X_bin, y_bin)


def test_internal_attrs_are_saved(pipeline):
    """Assert that cols and train_only attrs are stored after clone."""
    pl = pipeline(model=False)
    assert pl.steps[-1][1]._cols == ["mean radius"]
    assert pl.steps[-2][1]._train_only is True


def test_transform_only_X_or_y():
    """Assert that the pipeline can transform only X or y."""
    pl = Pipeline([("encoder", LabelEncoder()), ("scaler", StandardScaler())])
    pl.fit(X_bin, y_bin)
    assert isinstance(pl.transform(y=y_bin), pd.Series)
    assert isinstance(pl.transform(X=X_bin), pd.DataFrame)


def test_X_is_required_and_not_provided():
    """Assert that an error is raised when the transformer requires features."""
    pl = Pipeline(steps=[("scaler", StandardScaler())])
    with pytest.raises(ValueError, match=".*X is required but has not been provided.*"):
        pl.fit()  # StandardScaler.fit requires X


def test_X_is_required_but_provided_empty():
    """Assert that an error is raised when the transformer requires features."""
    pl = Pipeline(steps=[("scaler", StandardScaler())])
    with pytest.raises(ValueError, match=".*the provided feature set is empty.*"):
        pl.fit(pd.DataFrame())


def test_fit_transform(pipeline):
    """Assert that the pipeline can be fit-transformed normally."""
    pl = pipeline(model=False)
    pl.steps[0] = ("test", "passthrough")
    assert isinstance(pl.fit_transform(X_bin), pd.DataFrame)  # Returns X
    pl.steps[-1] = ("test_final", "passthrough")
    assert isinstance(pl.fit_transform(X_bin, y_bin), tuple)  # Returns X, y


def test_transform(pipeline):
    """Assert that the pipeline uses transform normally."""
    pl = pipeline(model=False)
    assert isinstance(pl.transform(X_bin), pd.DataFrame)
    assert isinstance(pl.transform(X_bin, y_bin), tuple)


def test_transform_both_None(pipeline):
    """Assert that an error is raised when both X and y are None."""
    pl = pipeline(model=False)
    with pytest.raises(ValueError, match=".*X and y cannot be both None.*"):
        pl.transform()


def test_transform_train_only(pipeline):
    """Assert that the pipeline ignores train_only during predicting."""
    pl = pipeline(model=False)
    assert len(pl.transform(X_bin)) == len(X_bin)  # Pruner is not applied


def test_inverse_transform_both_None(pipeline):
    """Assert that an error is raised when both X and y are None."""
    pl = pipeline(model=False)
    with pytest.raises(ValueError, match=".*X and y cannot be both None.*"):
        pl.inverse_transform()


def test_inverse_transform():
    """Assert that the pipeline uses inverse_transform normally."""
    pl = Pipeline([("scaler", StandardScaler())]).fit(X_bin)
    X = pl.inverse_transform(pl.transform(X_bin))
    assert_frame_equal(X_bin, X)


def test_decision_function(pipeline):
    """Assert that the pipeline uses decision_function normally."""
    pl = pipeline(model=True)
    assert isinstance(pl.decision_function(X_bin), np.ndarray)


def test_predict_no_parameters(pipeline_ts):
    """Assert that an error is raised when X and fh are both None."""
    with pytest.raises(ValueError, match=".*cannot be both None.*"):
        pipeline_ts.predict()


def test_predict(pipeline):
    """Assert that the pipeline uses predict normally."""
    pl = pipeline(model=True)
    assert isinstance(pl.predict(X_bin), np.ndarray)


def test_predict_ts_no_fh(pipeline_ts):
    """Assert that the pipeline uses predict for forecast."""
    with pytest.raises(ValueError, match=".*fh parameter cannot be None.*"):
        pipeline_ts.predict(range(3))


def test_predict_ts(pipeline_ts):
    """Assert that the pipeline uses predict for forecast."""
    assert isinstance(pipeline_ts.predict(fh=range(3)), pd.Series)


def test_predict_interval(pipeline_ts):
    """Assert that the pipeline uses predict_interval normally."""
    assert isinstance(pipeline_ts.predict_interval(fh=range(3)), pd.DataFrame)


def test_predict_log_proba(pipeline):
    """Assert that the pipeline uses predict_log_proba normally."""
    pl = pipeline(model=True)
    assert isinstance(pl.predict_log_proba(X_bin), np.ndarray)


def test_predict_proba_no_parameters(pipeline_ts):
    """Assert that an error is raised when X and fh are both None."""
    with pytest.raises(ValueError, match=".*cannot be both None.*"):
        pipeline_ts.predict_proba()


def test_predict_proba(pipeline):
    """Assert that the pipeline uses predict_proba normally."""
    pl = pipeline(model=True)
    assert isinstance(pl.predict_proba(X_bin), np.ndarray)


def test_predict_proba_ts_no_fh(pipeline_ts):
    """Assert that the pipeline uses predict for forecast."""
    with pytest.raises(ValueError, match=".*fh parameter cannot be None.*"):
        pipeline_ts.predict_proba(range(3))


def test_predict_proba_ts(pipeline_ts):
    """Assert that the pipeline uses predict_proba for forecast tasks."""
    assert isinstance(pipeline_ts.predict_proba(fh=range(3)), Normal)


def test_predict_quantiles(pipeline_ts):
    """Assert that the pipeline uses predict_quantiles normally."""
    assert isinstance(pipeline_ts.predict_quantiles(fh=range(3)), pd.DataFrame)


def test_predict_residuals(pipeline_ts):
    """Assert that the pipeline uses predict_residuals normally."""
    assert isinstance(pipeline_ts.predict_residuals(y=y_fc), pd.Series)


def test_predict_var(pipeline_ts):
    """Assert that the pipeline uses predict_var normally."""
    assert isinstance(pipeline_ts.predict_var(fh=range(3)), pd.DataFrame)


def test_score_no_parameters(pipeline_ts):
    """Assert that an error is raised when X and fh are both None."""
    with pytest.raises(ValueError, match=".*cannot be both None.*"):
        pipeline_ts.score()


def test_score(pipeline):
    """Assert that the pipeline uses score normally."""
    pl = pipeline(model=True)
    assert isinstance(pl.score(X_bin, y_bin), float)


def test_score_ts(pipeline_ts):
    """Assert that the pipeline uses score for forecast tasks."""
    assert isinstance(pipeline_ts.score(y=y_fc, fh=y_fc.index), float)

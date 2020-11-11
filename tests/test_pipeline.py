# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for pipeline.py

"""

# Standard packages
import pytest

# Own modules
from atom import ATOMClassifier
from .utils import X_bin, y_bin


# Test __init__ ============================================================ >>

def test_estimators_to_empty_series():
    """Assert that when starting atom, the estimators are empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.pipeline.estimators.empty


def test_input_is_copied():
    """Assert that the parameters are copied to attributes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline = "pipe_2"
    assert atom.pipeline.estimators is not atom._branches["main"].estimators
    assert atom.pipeline.data is not atom._branches["main"].data
    assert atom.pipeline.idx is not atom._branches["main"].idx
    assert atom.pipeline.mapping is not atom._branches["main"].mapping


# Test __repr__ ============================================================ >>

def test_repr():
    """Assert that when starting atom, the estimators are empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert str(atom.pipeline).startswith("Pipeline: main")


# Test rename ============================================================== >>

def test_rename_invalid_name():
    """Assert that an error is raised when name is empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=r".*A pipeline can't have an empty name!.*"):
        atom.pipeline.rename("")


def test_rename_method():
    """Assert that we can rename a pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline.rename("pipe_1")
    assert atom.pipeline.name == "pipe_1"
    assert atom.pipeline.estimators.name == "pipe_1"


# Test clear =============================================================== >>

def test_pipeline_clear():
    """Assert that we can clear a pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline = "pipe_2"
    atom.pipeline.clear()
    assert atom.pipelines == ["main"]


def test_pipeline_clear_invalid():
    """Assert that an error is raised when we try to clear the last pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(PermissionError, atom.pipeline.clear)

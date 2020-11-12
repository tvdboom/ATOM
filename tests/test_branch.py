# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for branch.py

"""

# Standard packages
import pytest

# Own modules
from atom import ATOMClassifier
from .utils import X_bin, y_bin


# Test __init__ ==================================================== >>

def test_estimators_to_empty_series():
    """Assert that when starting atom, the estimators are empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.estimators.empty


def test_input_is_copied():
    """Assert that the parameters are copied to attributes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "branch_2"
    assert atom.branch.estimators is not atom._branches["main"].estimators
    assert atom.branch.data is not atom._branches["main"].data
    assert atom.branch.idx is not atom._branches["main"].idx
    assert atom.branch.mapping is not atom._branches["main"].mapping


# Test __repr__ ==================================================== >>

def test_repr():
    """Assert that the __repr__  method returns the list of available branches."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "branch_2"
    assert str(atom.branch) == "Branches:\n --> main\n --> branch_2 !"


# Test status ====================================================== >>

def test_status_method():
    """Assert that the status method prints the estimators without errors."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch.status()


# Test rename ====================================================== >>

def test_rename_invalid_name():
    """Assert that an error is raised when name is empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=r".*A branch can't have an empty name!.*"):
        atom.branch.rename("")


def test_rename_method():
    """Assert that we can rename a branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch.rename("branch_1")
    assert atom.branch.name == "branch_1"
    assert atom.branch.estimators.name == "branch_1"


# Test delete ====================================================== >>

def test_branch_delete():
    """Assert that we can delete a branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "branch_2"
    atom.branch.delete()
    assert "branch_2" not in atom._branches


def test_branch_clear_invalid():
    """Assert that an error is raised when we try to clear the last branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(PermissionError, atom.branch.delete)

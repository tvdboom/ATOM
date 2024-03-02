"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for the utils module.

"""

from datetime import timedelta
from unittest.mock import patch

import pytest

from atom import show_versions
from atom.utils.utils import ClassMap, time_to_str


# Test _show_versions ============================================== >>

@patch.dict("sys.modules", {"sklearn": "1.3.2"}, clear=True)
def test_show_versions():
    """Assert that the show_versions function runs without errors."""
    show_versions()


# Test utils ======================================================= >>

def test_classmap_failed_initialization():
    """Assert that an error is raised when the classes do not have the key attribute."""
    with pytest.raises(ValueError, match=".*has no attribute.*"):
        ClassMap(2, 3)


def test_classmap_manipulations():
    """Assert that the ClassMap class can be manipulated."""
    cm = ClassMap(2, 3, 4, 5, 6, key="real")
    assert str(cm[[3, 4]]) == "[3, 4]"
    assert str(cm[3:5]) == "[5, 6]"
    assert str(cm[3]) == "3"
    cm[2] = 8
    assert str(cm) == "[2, 3, 8, 5, 6]"
    del cm[4]
    assert str(cm) == "[2, 3, 8, 5]"
    assert str(list(reversed(cm))) == "[5, 8, 3, 2]"
    cm += [6]
    assert str(cm) == "[2, 3, 8, 5, 6]"
    assert cm.index(3) == 1
    cm.clear()
    assert str(cm) == "[]"


def test_time_to_string():
    """Assert that the time strings are formatted properly."""
    assert time_to_str(timedelta(seconds=17).total_seconds()).startswith("17.00")
    assert time_to_str(timedelta(minutes=1, seconds=2).total_seconds()) == "01m:02s"
    assert time_to_str(timedelta(hours=3, minutes=8).total_seconds()) == "03h:08m:00s"

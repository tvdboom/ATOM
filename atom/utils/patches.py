"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing patches for external libraries.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from sklearn.model_selection._validation import _fit_and_score, _score

from atom.utils.types import Float


def fit_and_score(*args, **kwargs) -> dict[str, Any]:
    """Wrap sklearn's _fit_and_score function.

    Wrap the function sklearn.model_selection._validation._fit_and_score
    to, in turn, path sklearn's _score function to accept pipelines that
    drop samples during transforming, within a joblib parallel context.

    """

    def wrapper(*args, **kwargs) -> dict[str, Any]:
        with patch("sklearn.model_selection._validation._score", score(_score)):
            return _fit_and_score(*args, **kwargs)

    return wrapper(*args, **kwargs)


def score(f: Callable) -> Callable:
    """Patch decorator for sklearn's _score function.

    Monkey patch for sklearn.model_selection._validation._score
    function to score pipelines that drop samples during transforming.

    """

    def wrapper(*args, **kwargs) -> Float | dict[str, Float]:
        args_c = list(args)  # Convert to a list for item assignment
        if len(args[0]) > 1:  # Has transformers
            args_c[1], args_c[2] = args_c[0][:-1].transform(args_c[1], args_c[2])

        # Return f(final_estimator, X_transformed, y_transformed, ...)
        return f(args_c[0][-1], *args_c[1:], **kwargs)

    return wrapper

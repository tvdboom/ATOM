# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the Branch class.

"""

# Standard packages
import pandas as pd
from inspect import signature
from copy import copy, deepcopy
from typeguard import typechecked

# Own modules
from .basetransformer import BaseTransformer
from .utils import flt


class Branch(object):
    """Contains all information corresponding to a branch.

    Parameters
    ----------
    *args: arguments
        Parent class from which the branch is called and name of
        the branch.

    estimators: pd.Series or None, optional (default=None)
        Sequence of estimators fitted on the data in the branch.

    data: pd.DataFrame or None, optional (default=None)
        Dataset coupled to the branch.

    idx: tuple or None, optional (default=None)
        Tuple indicating the train and test sizes.

    mapping: dict or None, optional (default=None)
        Dictionary of the target values mapped to their respective
        encoded integer.

    feature_importance: list, optional (default=None)
        Features ordered by most to least important.

    """

    def __init__(
            self,
            *args,
            estimators=None,
            data=None,
            idx=None,
            mapping=None,
            feature_importance=None
    ):
        # Make copies of the parameters to not overwrite mutable variables
        self.T, self.name = args[0], args[1]
        if estimators is None:
            self.estimators = pd.Series([], name=self.name, dtype="object")
        else:
            self.estimators = copy(estimators)
        self.data = deepcopy(data)
        self.idx = copy(idx)
        self.mapping = copy(mapping)
        self.feature_importance = copy(feature_importance)

    def __repr__(self):
        out = "Branches:"
        for branch in self.T._branches.keys():
            out += f"\n --> {branch}"
            if branch == self.T._current:
                out += " !"

        return out

    def status(self):
        """Print the status of the branch."""
        self.T.log(f"Branch: {self.name}")
        for est in self.estimators:
            self.T.log(f" --> {est.__class__.__name__}")
            for param in signature(est.__init__).parameters:
                if param not in BaseTransformer.attrs + ["self"]:
                    self.T.log(f"   >>> {param}: {str(flt(getattr(est, param)))}")

    @typechecked
    def rename(self, name: str):
        """Change the name of the branch."""
        if not name:
            raise ValueError("A branch can't have an empty name!")
        else:
            self.name = name
            self.estimators.name = name
            self.T._branches[name] = self.T._branches.pop(self.T._current)
            self.T._current = name
            self.T.log("Branch renamed successfully!", 1)

    def delete(self):
        """Remove the current branch."""
        if len(self.T._branches) > 1:
            self.T._branches.pop(self.T._current)  # Delete the pipeline
            self.T._current = list(self.T._branches.keys())[0]  # Reset the current one
            self.T.log("Branch deleted successfully!")
        else:
            raise PermissionError("Can't delete the last branch in the pipeline!", 1)

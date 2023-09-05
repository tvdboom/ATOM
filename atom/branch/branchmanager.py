# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BranchManager class.

"""

from __future__ import annotations

import re
from copy import copy
from functools import cached_property
from typing import Hashable

import dill
import pandas as pd
from beartype import beartype
from joblib.memory import Memory
from sklearn.utils.validation import check_memory

from atom.branch.branch import Branch
from atom.utils.types import (
    BOOL, BRANCH, DATAFRAME, DATAFRAME_TYPES, FEATURES, INDEX, INT, INT_TYPES,
    PANDAS, SEQUENCE, SERIES_TYPES, SLICE, TARGET,
)
from atom.utils.utils import (
    CustomDict, DataContainer, bk, custom_transform, flt, get_cols, lst, merge,
    to_pandas, ClassMap, DataContainer
)


@beartype
class BranchManager:
    """Manages branches.

    Parameters
    ----------
    memory: Memory
        Location to store inactive branches. If None, all branches
        are kept in memory.

    """

    def __init__(self, memory: Memory):
        self.memory = memory

        self.branches = ClassMap()
        self.add(Branch("main"))

        self.holdout = None

    def __str__(self) -> str:
        return f"BranchManager({', '.join([b.name for b in self.branches])})"

    def __contains__(self, item: str) -> BOOL:
        return item in self.branches

    def __getitem__(self, item: str) -> Branch:
        if item in self.branches:
            return self.branches[item]
        else:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance has no branch {item}."
            )

    @property
    def og(self) -> Branch:
        """Branch containing the original dataset.

        This branch contains the data prior to any transformations.
        It redirects to the current branch if its pipeline is empty
        to not have the same data in memory twice.

        """
        return self.branches.get("og", self.current.name)

    @property
    def current(self) -> Branch:
        return self._current

    @current.setter
    def current(self, branch: str):
        self.store()
        self._current = self.branches[branch]
        self.load()

    def add(self, name: str, parent: Branch):
        """

        Parameters
        ----------
        name
        parent

        """
        if name not in self.branches:
            if self.branches:
                self.store()

            self._current = self.branches.append(Branch(name))
            self._current.pipeline.memory = self.memory

            if parent:
                # Copy the data attrs (except holdout) and point to the rest
                for attr in ("_data", "_pipeline", "_mapping"):
                    setattr(self.current, attr, copy(getattr(parent, attr)))
                for attr in vars(parent):
                    if not hasattr(self.current, attr):  # If not already assigned...
                        setattr(self.current, attr, getattr(parent, attr))

    def fill(self, data: DataContainer, holdout: DATAFRAME | None = None):
        """Fill the current branch with data.

        Parameters
        ----------
        data: DataContainer
            New data for the current branch.

        holdout: dataframe or None, default=None
            Holdout data set (if any).

        """
        self.current.data = data
        self.holdout = self.current.holdout = holdout

    def store(self):
        """Store the branch's data as a pickle in memory.

        After storage, the DataContainer is deleted and the branch is
        no longer usable until `load` is called. This method is used
        for inactive branches.

        Parameters
        ----------
        branch: Branch
            Path to the storage location.

        """
        if self.memory.location is not None:
            with open(f"{self.memory.location}Branch({self.current.name})", "wb") as file:
                dill.dump(self.current._data, file)

            self.current._data = None

    def load(self, branch: Branch) -> Branch:
        """Load the branch's data from memory.

        This method is used to restore the data of inactive branches.

        """
        if self.location is not None:
            with open(f"{self.memory.location}Branch({branch.name})", "rb") as file:
                branch._data = dill.load(file)

        return branch

# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BranchManager class.

"""

from __future__ import annotations

from copy import copy, deepcopy

from beartype import beartype
from joblib.memory import Memory
from sklearn.utils.validation import check_memory

from atom.branch.branch import Branch
from atom.utils.types import BOOL, DATAFRAME, INT, SEQUENCE
from atom.utils.utils import ClassMap, DataContainer


@beartype
class BranchManager:
    """Manages branches.

    Parameters
    ----------
    memory: Memory
        Location to store inactive branches. If None, all branches
        are kept in memory.

    """

    def __init__(
        self,
        branches: SEQUENCE | None = None,
        og: Branch | None = None,
        memory: Memory | None = None,
    ):
        self.memory = check_memory(memory)

        if branches is None:
            self.branches = ClassMap()
            self.add("main")
        else:
            self.branches = ClassMap(*list(branches))

        self._og = og

    def __str__(self) -> str:
        return f"BranchManager({', '.join(self.branches.keys())})"

    def __len__(self) -> INT:
        return len(self.branches)

    def __iter__(self):
        yield from self.branches

    def __contains__(self, item: str) -> BOOL:
        return item in self.branches

    def __getitem__(self, item: INT | str) -> Branch:
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
        return self._og or self.current

    @property
    def current(self) -> Branch:
        return self._current

    @current.setter
    def current(self, branch: str):
        self._current.store()
        self._current = self.branches[branch]
        self._current.load()

    @current.deleter
    def current(self):
        del self.branches[self.current.name]
        self._current = self.branches[0]

    @staticmethod
    def _copy_from_parent(branch: Branch, parent: Branch):
        """Pass data and attributes from a parent to a new branch.

        Parameters
        ----------
        branch: Branch
            Pass data and attributes to this branch.

        parent: Branch
            Parent branch from which to get the info from.

        """
        # Transfer data from parent or load from memory
        if parent._data is None:
            setattr(branch, "_data", parent.load(assign=False))
        else:
            setattr(branch, "_data", deepcopy(parent._data))

        setattr(branch, "_pipeline", copy(getattr(parent, "_pipeline")))
        setattr(branch, "_mapping", copy(getattr(parent, "_mapping")))
        for attr in vars(parent):
            if not hasattr(branch, attr):  # If not already assigned...
                setattr(branch, attr, getattr(parent, attr))

    def add(self, name: str, parent: Branch | None = None):
        """Add a new branch to the manager.

        If the branch is called `og` (reserved name for the original
        branch), it's created separately and stored in memory.

        Parameters
        ----------
        name: str
            Name for the new branch.

        parent: Branch or None, default=None
            Parent branch. Data and attributes from the parent are
            passed to the new branch.

        """
        if name == "og":
            if not self._og:
                self._og = Branch("og", memory=self.memory)
                self._copy_from_parent(self._og, self.current)
        else:
            # Skip for first call from __init__
            if self.branches:
                self.current.store()

            self._current = self.branches.append(Branch(name, memory=self.memory))

            if parent is not None:
                self._copy_from_parent(self._current, parent)

    def fill(self, data: DataContainer, holdout: DATAFRAME | None = None):
        """Fill the current branch with data.

        Parameters
        ----------
        data: DataContainer
            New data for the current branch.

        holdout: dataframe or None, default=None
            Holdout data set (if any).

        """
        self.current._data = data
        self.current._holdout = holdout

    def reset(self):
        """Reset this instance to its initial state."""
        self.branches = ClassMap()
        self.add("main", parent=self.og)
        self._og = None

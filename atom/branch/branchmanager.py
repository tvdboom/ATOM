# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the BranchManager class.

"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from copy import copy, deepcopy

from beartype import beartype
from joblib.memory import Memory
from sklearn.utils.validation import check_memory

from atom.branch.branch import Branch
from atom.utils.types import Bool, DataFrame, Int
from atom.utils.utils import ClassMap, DataContainer


@beartype
class BranchManager:
    """Object that manages branches.

    Maintains references to a series of branches and the current
    active branch. Additionally, always stores an 'original' branch
    containing the original dataset (previous to any transformations).
    The branches share a reference to a holdout set, not the instance
    self. When a memory object is specified, it stores inactive
    branches in memory.

    Read more in the [user guide][branches].

    !!! warning
        This class should not be called directly. The BranchManager is
        created internally by the [ATOMClassifier][], [ATOMForecaster][]
        and [ATOMRegressor][] classes.

    Parameters
    ----------
    memory: str, [Memory][joblibmemory] or None, default=None
        Location to store inactive branches. If None, all branches
        are kept in memory. This memory object is passed to the
        branches for pipeline caching.

    Attributes
    ----------
    branches: ClassMap
        Collection of branches.

    og: [Branch][]
        Branch containing the original dataset. It can be any branch
        in `branches` or an internally made branch called `og`.

    current: [Branch][]
        Current active branch.

    See Also
    --------
    atom.branch:Branch

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Initialize atom
    atom = ATOMClassifier(X, y, verbose=2)

    # Train a model
    atom.run("RF")

    # Change the branch and apply feature scaling
    atom.branch = "scaled"

    atom.scale()
    atom.run("RF_scaled")

    # Compare the models
    atom.plot_roc()
    ```

    """

    def __init__(self, memory: str | Memory | None = None):
        self.memory = check_memory(memory)

        self.branches = ClassMap()
        self.add("main")

        self._og: Branch | None = None

    def __repr__(self) -> str:
        return f"BranchManager([{', '.join(self.branches.keys())}], og={self.og.name})"

    def __len__(self) -> Int:
        return len(self.branches)

    def __iter__(self) -> Iterator[Branch]:
        yield from self.branches

    def __contains__(self, item: str) -> bool:
        return item in self.branches

    def __getitem__(self, item: Int | str) -> Branch:
        try:
            return self.branches[item]
        except KeyError:
            raise IndexError(
                f"This {self.__class__.__name__} instance has no branch {item}."
            )

    @property
    def og(self) -> Branch:
        """Branch containing the original dataset.

        This branch contains the data prior to any transformations.
        It redirects to the first branch with an empty pipeline (if
        it exists), or to an internally made branch called `og`.

        """
        return self._og or next(b for b in self.branches if not b.pipeline.steps)

    @property
    def current(self) -> Branch:
        """Current active branch."""
        return self._current

    @current.setter
    def current(self, branch: str):
        self._current.store()
        self._current: Branch = self.branches[branch]
        self._current.load()

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
        if branch.name == "og" and parent._location and branch._location:
            # Make a new copy of the data for the og branch
            parent.store(assign=False)
            shutil.copy(
                parent._location.joinpath(f"{parent}.pkl"),
                branch._location.joinpath(f"{branch}.pkl"),
            )
        elif parent._location:
            # Transfer data from memory to avoid having
            # the datasets in memory twice at one time
            parent.store()
            setattr(branch, "_container", parent.load(assign=False))
        else:
            # Copy the dataset in-memory
            setattr(branch, "_container", deepcopy(parent._container))

        # Deepcopy the pipeline but use the same estimators
        setattr(branch, "_pipeline", deepcopy(getattr(parent, "_pipeline")))
        for i, step in enumerate(parent._pipeline.steps):
            branch.pipeline.steps[i] = step

        # Copy mapping and assign other vars
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
                self._copy_from_parent(self.og, self.current)
        else:
            # Skip for first call from __init__
            if self.branches:
                self.current.store()

            self._current = self.branches.append(Branch(name, memory=self.memory))

            if parent:
                self._copy_from_parent(self.current, parent)

    def fill(self, data: DataContainer, holdout: DataFrame | None = None):
        """Fill the current branch with data.

        Parameters
        ----------
        data: DataContainer
            New data for the current branch.

        holdout: dataframe or None, default=None
            Holdout data set (if any).

        """
        self.current._container = data
        self.current._holdout = holdout

    def reset(self, hard: Bool = False):
        """Reset this instance to its initial state.

        The initial state of the BranchManager contains a single branch
        called `main` with no data. There's no reference to an original
        (`og`) branch.

        Parameters
        ----------
        hard: bool, default=False
            If True, flushes completely the cache.

        """
        self.branches = ClassMap()
        self.add("main", parent=self.og)
        self._og = None

        if hard:
            self.memory.clear(warn=False)

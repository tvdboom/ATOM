# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the BranchManager class.

"""

from __future__ import annotations

from copy import copy, deepcopy
import shutil
from beartype import beartype
from joblib.memory import Memory

from atom.branch.branch import Branch
from atom.utils.types import Bool, DataFrame, Int
from atom.utils.utils import ClassMap, DataContainer


@beartype
class BranchManager:
    """Object to manage branches.

    Maintains references to a series of branches and the current
    active branch. Additionally, stores an 'original' branch separately
    from the rest, intended for the branch containing the original
    dataset (previous to any transformations). The branches share a
    reference to any holdout set, not the BranchManager self.

    Read more in the [user guide][branches].

    Parameters
    ----------
    memory: [Memory][joblibmemory]
        Location to store inactive branches. If None, all branches
        are kept in memory.

    Attributes
    ----------
    branches: ClassMap
        Collection of branches.

    og: [Branch][] or None
        Original branch. None if no branches have transformations.

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

    def __init__(self, memory: Memory):
        self.memory = memory

        self.branches = ClassMap()
        self.add("main")

        self._og = None

    def __repr__(self) -> str:
        return f"BranchManager([{', '.join(self.branches.keys())}], og={self.og})"

    def __len__(self) -> Int:
        return len(self.branches)

    def __iter__(self):
        yield from self.branches

    def __contains__(self, item: str) -> Bool:
        return item in self.branches

    def __getitem__(self, item: Int | str) -> Branch:
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
        It redirects to the first branch with an empty pipeline to not
        have the same data in memory twice.

        """
        return self._og or next(b for b in self.branches if not b.pipeline.steps)

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
        if branch.name == "og" and parent._location:
            # Make a new copy of the data for the og branch
            parent.store(assign=False)
            shutil.copy(f"{parent._location}.pkl", f"{branch._location}.pkl")
        elif parent._location:
            # Transfer data from memory to avoid having
            # the datasets in memory twice at one time
            parent.store()
            setattr(branch, "_data", parent.load(assign=False))
        else:
            # Copy the dataset in-memory
            setattr(branch, "_data", deepcopy(parent._data))

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
        self.current._data = data
        self.current._holdout = holdout

    def reset(self, hard: Bool = False):
        """Reset this instance to its initial state.

        The initial state of the BranchManager contains a single branch
        called `main` with no data. There's no reference to an original
        (`og`) branch.

        Parameters
        ----------
        hard: bool, default=False
            If True, also deletes the cached memory (if enabled).

        """
        self.branches = ClassMap()
        self.add("main", parent=self.og)
        self._og = None

        if hard:
            self.pipeline._memory.clear()

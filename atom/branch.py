# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the Branch class.

"""

# Standard packages
import pandas as pd
from inspect import signature
from copy import copy
from typeguard import typechecked
from typing import Optional

# Own modules
from .basetransformer import BaseTransformer
from .utils import (
    X_TYPES, SEQUENCE_TYPES, flt, merge, to_df, to_series,
    check_deep, composed, crash, method_to_log,
)


class Branch:
    """Contains all information corresponding to a branch.

    Parameters
    ----------
    *args: arguments
        Parent class from which the branch is called and name of
        the branch.

    parent: str or None, optional (default=None)
        Name of the branch from which to split.

    Attributes
    ----------
    pipeline: pd.Series or None, optional (default=None)
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

    # Private attributes/properties (not accessible from atom)
    private = ("T", "name", "data", "idx")

    def __init__(self, *args, parent=None):
        # Make copies of the params to not overwrite mutable variables
        self.T, self.name = args[0], args[1]
        if not parent:
            self.pipeline = pd.Series([], name=self.name, dtype="object")
            self.data = None
            self.idx = None
            self.mapping = None
            self.feature_importance = None
        else:
            self.pipeline = copy(self.T._branches[parent].pipeline)
            self.data = copy(self.T._branches[parent].data)
            self.idx = copy(self.T._branches[parent].idx)
            self.mapping = copy(self.T._branches[parent].mapping)
            self.feature_importance = copy(self.T._branches[parent].feature_importance)

    def __repr__(self):
        out = "Branches:"
        for branch in self.T._branches.keys():
            out += f"\n --> {branch}"
            if branch == self.T._current:
                out += " !"

        return out

    # Utility methods ============================================== >>

    def _get_depending_models(self):
        """Return the models that are dependent on this branch."""
        return [m.name for m in self.T._models if m.branch.name == self.name]

    @composed(crash, method_to_log)
    def status(self):
        """Print the status of the branch."""
        self.T.log(f"Branch: {self.name}")

        # Add the transformers with their parameters
        self.T.log(f" --> Pipeline: {None if self.pipeline.empty else ''}")
        for est in self.pipeline:
            self.T.log(f"   >>> {est.__class__.__name__}")
            for param in signature(est.__init__).parameters:
                if param not in BaseTransformer.attrs + ["self"]:
                    self.T.log(f"     --> {param}: {str(flt(getattr(est, param)))}")

        # Add the models linked to the branch
        dependent = self._get_depending_models()
        self.T.log(f" --> Models: {', '.join(dependent) if dependent else None}")

    @composed(crash, method_to_log, typechecked)
    def rename(self, name: str):
        """Change the name of the current branch."""
        if not name:
            raise ValueError("A branch can't have an empty name!")
        else:
            self.name = name
            self.pipeline.name = name
            self.T._branches[name] = self.T._branches.pop(self.T._current)
            self.T.log(f"Branch '{self.T._current}' was renamed to '{name}'.", 1)
            self.T._current = name

    @composed(crash, method_to_log, typechecked)
    def delete(self, name: Optional[str] = None):
        """Remove a branch from the pipeline."""
        if name is None:
            name = self.T._current
        elif name not in self.T._branches:
            raise ValueError(f"Branch '{name}' not found in the pipeline!")

        dependent = self.T._branches[name]._get_depending_models()
        if len(self.T._branches) == 1:
            raise PermissionError("Can't delete the last branch in the pipeline!")
        elif len(dependent):
            raise PermissionError(
                "Can't delete a branch with depending models! Consider deleting "
                f"the models first. Depending models are: {', '.join(dependent)}."
            )
        else:
            self.T._branches.pop(name)
            if name == self.T._current:  # Reset the current branch
                self.T._current = list(self.T._branches.keys())[0]
            self.T.log(f"Branch '{name}' successfully deleted!")

    # Data properties ============================================== >>

    def _check_setter(self, name, value):
        """Check the property setter.

        Convert the property to a pandas object and compare with the
        rest of the dataset to check if it has the right dimensions.

        Parameters
        ----------
        name: str
            Name of the property to check.

        value: sequence, dict or pd.DataFrame
            Property to be checked.

        """
        def counter(name, dim):
            """Return the counter dimension of the provided data set."""
            if dim == "side":
                if "X" in name:
                    return name.replace("X", "y")
                if "y" in name:
                    return name.replace("y", "X")
            else:
                if "train" in name:
                    return name.replace("train", "test")
                if "test" in name:
                    return name.replace("test", "train")

        # Define the data attrs side and under
        side_name = counter(name, "side")
        side = getattr(self, side_name) if side_name else None
        under_name = counter(name, "under")
        under = getattr(self, under_name) if under_name else None

        # Convert (if necessary) to pandas
        if "y" in name:
            value = to_series(
                value,
                index=side.index if side_name else None,
                name=under.name if under_name else "target"
            )
        else:
            value = to_df(
                value,
                index=side.index if side_name else None,
                columns=under.columns if under_name else None
            )

        if side_name:  # Check for equal number of rows
            if len(value) != len(side):
                raise ValueError(
                    f"{name} and {side_name} must have the same "
                    f"number of rows, got {len(value)} != {len(side)}."
                )
            if not value.index.equals(side.index):
                raise ValueError(
                    f"{name} and {side_name} must have the same "
                    f"indices, got {value.index} != {side.index}."
                )

        if under_name:  # Check they have the same columns
            if "y" in name:
                if value.name != under.name:
                    raise ValueError(
                        f"{name} and {under_name} must have the "
                        f"same name, got {value.name} != {under.name}."
                    )
            else:
                if value.shape[1] != under.shape[1]:
                    raise ValueError(
                        f"{name} and {under_name} must have the same number "
                        f"of columns, got {value.shape[1]} != {under.shape[1]}."
                    )

                if list(value.columns) != list(under.columns):
                    raise ValueError(
                        f"{name} and {under_name} must have the same "
                        f"columns , got {value.columns} != {under.columns}."
                    )

        return value

    @property
    def dataset(self):
        """Complete data set."""
        return self.data

    @dataset.setter
    @typechecked
    def dataset(self, dataset: Optional[X_TYPES]):
        self.data = self._check_setter("dataset", dataset)

    @property
    def train(self):
        """Training set."""
        return self.data[:self.idx[0]]

    @train.setter
    @typechecked
    def train(self, train: X_TYPES):
        df = self._check_setter("train", train)
        self.data = pd.concat([df, self.test])
        self.idx[0] = len(df)

    @property
    def test(self):
        """Test set."""
        return self.data[-self.idx[1]:]

    @test.setter
    @typechecked
    def test(self, test: X_TYPES):
        df = self._check_setter("test", test)
        self.data = pd.concat([self.train, df])
        self.idx[1] = len(df)

    @property
    def X(self):
        """Feature set."""
        return self.data.drop(self.target, axis=1)

    @X.setter
    @typechecked
    def X(self, X: X_TYPES):
        df = self._check_setter("X", X)
        self.data = merge(df, self.y)

    @property
    def y(self):
        """Target column."""
        return self.data[self.target]

    @y.setter
    @typechecked
    def y(self, y: SEQUENCE_TYPES):
        series = self._check_setter("y", y)
        self.data = merge(self.data.drop(self.target, axis=1), series)

    @property
    def X_train(self):
        """Feature set of the training set."""
        return self.train.drop(self.target, axis=1)

    @X_train.setter
    @typechecked
    def X_train(self, X_train: X_TYPES):
        df = self._check_setter("X_train", X_train)
        self.data = pd.concat([merge(df, self.train[self.target]), self.test])

    @property
    def X_test(self):
        """Feature set of the test set."""
        return self.test.drop(self.target, axis=1)

    @X_test.setter
    @typechecked
    def X_test(self, X_test: X_TYPES):
        df = self._check_setter("X_test", X_test)
        self.data = pd.concat([self.train, merge(df, self.test[self.target])])

    @property
    def y_train(self):
        """Target column of the training set."""
        return self.train[self.target]

    @y_train.setter
    @typechecked
    def y_train(self, y_train: SEQUENCE_TYPES):
        series = self._check_setter("y_train", y_train)
        self.data = pd.concat([merge(self.X_train, series), self.test])

    @property
    def y_test(self):
        """Target column of the test set."""
        return self.test[self.target]

    @y_test.setter
    @typechecked
    def y_test(self, y_test: SEQUENCE_TYPES):
        series = self._check_setter("y_test", y_test)
        self.data = pd.concat([self.train, merge(self.X_test, series)])

    @property
    def shape(self):
        """Shape of the dataset (n_rows, n_cols) or (n_rows, shape_row, n_cols)."""
        if not check_deep(self.X):
            return self.data.shape
        else:
            return len(self.data), self.X.iloc[0, 0].shape, 2

    @property
    def columns(self):
        """Name of all the columns."""
        return list(self.data.columns)

    @property
    def n_columns(self):
        """Number of columns."""
        return len(self.columns)

    @property
    def features(self):
        """Name of the features."""
        return self.columns[:-1]

    @property
    def n_features(self):
        """Number of features."""
        return len(self.features)

    @property
    def target(self):
        """Name of the target column."""
        return self.columns[-1]

    @property
    def classes(self):
        """Number of samples per class and per data set."""
        df = pd.DataFrame({
            "dataset": self.y.value_counts(sort=False, dropna=False),
            "train": self.y_train.value_counts(sort=False, dropna=False),
            "test": self.y_test.value_counts(sort=False, dropna=False),
        }, index=self.mapping.values())

        return df.fillna(0)  # If 0 counts, it doesnt return the row (gets a NaN)

    @property
    def n_classes(self):
        """Number of classes in the target column."""
        return len(self.y.unique())

# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
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
from .models import MODELS_ENSEMBLES
from .utils import (
    X_TYPES, SEQUENCE_TYPES, flt, merge, to_df, to_series,
    is_multidim, composed, crash, method_to_log,
)


class Branch:
    """Contains all information corresponding to a branch.

    Parameters
    ----------
    *args
        Parent class (from which the branch is called) and name.

    parent: str, Branch or None, optional (default=None)
        Name or branch from which to split. If None, create an
        empty branch.

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

    def __init__(self, *args, parent=None):
        # Make copies of the params to not overwrite mutable variables
        self.T, self.name = args[0], args[1]
        if not parent:
            self.pipeline = pd.Series(data=[], name=self.name, dtype="object")
            for attr in ("data", "idx", "mapping", "feature_importance"):
                setattr(self, attr, None)
        else:
            if isinstance(parent, str):
                parent = self.T._branches[parent]

            # Copy the branch attrs and point to the rest
            for attr in ("pipeline", "data", "idx", "mapping", "feature_importance"):
                setattr(self, attr, copy(getattr(parent, attr)))
            for attr in vars(parent):
                if not hasattr(self, attr):  # If not already assigned...
                    setattr(self, attr, getattr(parent, attr))

    def __repr__(self):
        out = f"Branch: {self.name}"

        # Add the transformers with their parameters
        out += f"\n --> Pipeline: {None if self.pipeline.empty else ''}"
        for est in self.pipeline:
            out += f"\n   >>> {est.__class__.__name__}"
            for param in signature(est.__init__).parameters:
                if param not in BaseTransformer.attrs + ["self"]:
                    out += f"\n     --> {param}: {str(flt(getattr(est, param)))}"

        # Add the models linked to the branch
        dependent = self._get_depending_models()
        out += f"\n --> Models: {', '.join(dependent) if dependent else None}"

        return out

    # Utility methods ============================================== >>

    def _get_attrs(self):
        """Get properties and attributes to call from parent."""
        props = [p for p in dir(self) if isinstance(getattr(Branch, p, None), property)]
        attrs = [a for a in vars(self) if a not in ("T", "name", "data", "idx")]
        return props + attrs

    def _get_depending_models(self):
        """Return the models that are dependent on this branch."""
        return [m.name for m in self.T._models.values() if m.branch is self]

    @composed(crash, method_to_log, typechecked)
    def delete(self, name: Optional[str] = None):
        """Delete the branch and all the models in it."""
        if name is None:
            name = self.name

        if name == "og":
            raise PermissionError(
                "The og branch is an internal branch and can not be deleted!"
            )
        elif name not in self.T._branches:
            raise ValueError(f"Branch {name} not found in {self.T.__class__.__name__}!")
        elif len(self.T._branches) <= 2:
            raise PermissionError(
                f"Can't delete the last branch in {self.T.__class__.__name__}!"
            )
        else:
            branch = self.T._branches[name]

            # Delete all depending models
            depending_models = branch._get_depending_models()
            if depending_models:
                self.T.delete(depending_models)

            # Reset the current branch
            if branch.name == self.T._current:
                self.T._current = list(self.T._branches)[1]  # 0 is og

            self.T._branches.pop(branch.name)
            self.T.log(f"Branch {branch.name} successfully deleted.", 1)

    @composed(crash, method_to_log, typechecked)
    def rename(self, name: str):
        """Change the name of the branch."""
        if not name:
            raise ValueError("A branch can't have an empty name!")
        elif name in self.T._branches:
            raise ValueError(f"Branch {self.T._branches[name].name} already exists!")
        else:
            for model in MODELS_ENSEMBLES.values():
                if name.lower().startswith(model.acronym.lower()):
                    raise ValueError(
                        "Invalid name for the branch. The name of a branch may "
                        f"not begin with a model's acronym, and {model.acronym} "
                        f"is the acronym of the {model.fullname} model."
                    )

        self.name = name
        self.pipeline.name = name
        self.T._branches[name] = self.T._branches.pop(self.T._current)
        self.T.log(f"Branch {self.T._current} is renamed to {name}.", 1)
        self.T._current = name

    @composed(crash, method_to_log)
    def status(self):
        """Get an overview of the pipeline and models in the branch."""
        self.T.log(str(self))

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

        if self._get_depending_models():
            raise PermissionError(
                "It's not allowed to change the data in the branch "
                "after it has been used to train models. Create a "
                "new branch to continue the pipeline."
            )

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
                name=under.name if under_name else "target",
            )
        else:
            value = to_df(
                value,
                index=side.index if side_name else None,
                columns=under.columns if under_name else None,
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

        if under_name:  # Check for equal number of columns
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
    def dataset(self, value: X_TYPES):
        self.data = value

    @property
    def train(self):
        """Training set."""
        return self.data[:self.idx[0]]

    @train.setter
    @typechecked
    def train(self, value: X_TYPES):
        df = self._check_setter("train", value)
        self.data = pd.concat([df, self.test])
        self.idx[0] = len(df)

    @property
    def test(self):
        """Test set."""
        return self.data[-self.idx[1]:]

    @test.setter
    @typechecked
    def test(self, value: X_TYPES):
        df = self._check_setter("test", value)
        self.data = pd.concat([self.train, df])
        self.idx[1] = len(df)

    @property
    def X(self):
        """Feature set."""
        return self.data.drop(self.target, axis=1)

    @X.setter
    @typechecked
    def X(self, value: X_TYPES):
        df = self._check_setter("X", value)
        self.data = merge(df, self.y)

    @property
    def y(self):
        """Target column."""
        return self.data[self.target]

    @y.setter
    @typechecked
    def y(self, value: SEQUENCE_TYPES):
        series = self._check_setter("y", value)
        self.data = merge(self.data.drop(self.target, axis=1), series)

    @property
    def X_train(self):
        """Features of the training set."""
        return self.train.drop(self.target, axis=1)

    @X_train.setter
    @typechecked
    def X_train(self, value: X_TYPES):
        df = self._check_setter("X_train", value)
        self.data = pd.concat([merge(df, self.train[self.target]), self.test])

    @property
    def X_test(self):
        """Features of the test set."""
        return self.test.drop(self.target, axis=1)

    @X_test.setter
    @typechecked
    def X_test(self, value: X_TYPES):
        df = self._check_setter("X_test", value)
        self.data = pd.concat([self.train, merge(df, self.test[self.target])])

    @property
    def y_train(self):
        """Target column of the training set."""
        return self.train[self.target]

    @y_train.setter
    @typechecked
    def y_train(self, value: SEQUENCE_TYPES):
        series = self._check_setter("y_train", value)
        self.data = pd.concat([merge(self.X_train, series), self.test])

    @property
    def y_test(self):
        """Target column of the test set."""
        return self.test[self.target]

    @y_test.setter
    @typechecked
    def y_test(self, value: SEQUENCE_TYPES):
        series = self._check_setter("y_test", value)
        self.data = pd.concat([self.train, merge(self.X_test, series)])

    @property
    def shape(self):
        """Shape of the dataset (n_rows, n_cols) or (n_rows, shape_row, n_cols)."""
        if not is_multidim(self.X):
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

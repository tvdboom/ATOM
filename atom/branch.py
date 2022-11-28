# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the Branch class.

"""

from copy import copy
from typing import Optional, Tuple, Union

import pandas as pd
from typeguard import typechecked

from atom.models import MODELS_ENSEMBLES
from atom.utils import (
    PANDAS_TYPES, SEQUENCE_TYPES, X_TYPES, CustomDict, composed, crash,
    custom_transform, merge, method_to_log, to_df, to_series,
)


class Branch:
    """Contains information corresponding to a branch.

    A branch contains a specific pipeline, the dataset transformed
    through that pipeline, the models fitted on that dataset, and all
    data and utility attributes that refer to that dataset. Branches
    can be created and accessed through atom's `branch` attribute.

    All properties and attributes of the branch (except the private
    ones, starting with underscore) can be accessed from the parent.

    Parameters
    ----------
    *args
        Class from which the branch is created and name of the branch.

    parent: Branch or None, default=None
        Branch from which to split. If None, create an empty branch.

    Attributes
    ----------
    name: str
        Name of the branch.

    """

    def __init__(self, *args, parent=None):
        self.T = args[0]

        self._name = args[1]
        self._parent = self.T._current
        self._pipeline = pd.Series(data=[], name=self.name, dtype="object")
        self._mapping = CustomDict()

        self._data = None
        self._idx = None
        self._holdout = None

        # If a parent branch is provided, copy its attrs to this one
        # _holdout is always reset since it wouldn't recalculate if
        # changes were made to the pipeline
        if parent:
            self._parent = parent.name

            # Copy the branch attrs and point to the rest
            for attr in ("_data", "_idx", "_pipeline", "_mapping"):
                setattr(self, attr, copy(getattr(parent, attr)))
            for attr in vars(parent):
                if not hasattr(self, attr):  # If not already assigned...
                    setattr(self, attr, getattr(parent, attr))

    def __delete__(self, instance):
        if len(self.T._branches.min("og")) == 1:
            raise PermissionError("Can't delete the last branch!")
        else:
            # Delete all depending models
            if depending_models := instance._get_depending_models():
                self.T.delete(depending_models)

            # If this is the last og branch, create a new one
            if self.T._get_og_branches() == [instance]:
                self.T._branches.insert(0, "og", Branch(self.T, "og", parent=self))

            # Reset the current branch
            if instance.name == self.T._current:
                self.T._current = list(self.T._branches.min("og"))[0]

            self.T._branches.pop(instance.name)
            self.T.log(f"Branch {instance.name} successfully deleted.", 1)
            self.T.log(f"Switched to branch {self.T._current}.", 1)

    def __repr__(self) -> str:
        out = f"Branch: {self.name}"

        # Add the transformers with their parameters
        out += f"\n --> Pipeline: {None if self.pipeline.empty else ''}"
        for est in self.pipeline:
            out += f"\n   --> {est.__class__.__name__}"

        # Add the models linked to the branch
        dependent = self._get_depending_models()
        out += f"\n --> Models: {', '.join(dependent) if dependent else None}"

        return out

    @property
    def name(self) -> str:
        """Branch's name."""
        return self._name

    @name.setter
    @typechecked
    def name(self, value: str):
        if not value:
            raise ValueError("A branch can't have an empty name!")
        elif value in self.T._branches:
            raise ValueError(f"Branch {self.T._branches[value].name} already exists!")
        else:
            for model in MODELS_ENSEMBLES.values():
                if value.lower().startswith(model.acronym.lower()):
                    raise ValueError(
                        "Invalid name for the branch. The name of a branch may "
                        f"not begin with a model's acronym, and {model.acronym} "
                        f"is the acronym of the {model._fullname} model."
                    )

        self._name = value
        self.pipeline.name = value
        self.T._branches[value] = self.T._branches.pop(self.T._current)
        self.T.log(f"Branch {self.T._current} is renamed to {value}.", 1)
        self.T._current = value

    # Data properties ============================================== >>

    def _check_setter(
        self,
        name: str,
        value: Union[SEQUENCE_TYPES, X_TYPES],
    ) -> PANDAS_TYPES:
        """Check the property setter.

        Convert the property to a pandas object and compare with the
        rest of the dataset to check if it has the right dimensions.

        Parameters
        ----------
        name: str
            Name of the property to check.

        value: sequence or dataframe-like
            New values for the property.

        Returns
        -------
        pd.Series or pd.DataFrame
            Data set.

        """

        def counter(name, dim):
            """Return the counter dimension of the provided data set."""
            if name == "dataset":
                return name
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
        if side_name:
            side = getattr(self, side_name)
        under_name = counter(name, "under")
        if under_name:
            under = getattr(self, under_name)

            # Convert (if necessary) to pandas
        if "y" in name:
            value = to_series(
                data=value,
                index=side.index if side_name else None,
                name=under.name if under_name else "target",
                dtype=under.dtype if under_name else None,
            )
        else:
            value = to_df(
                data=value,
                index=side.index if side_name else None,
                columns=under.columns if under_name else None,
                dtypes=under.dtypes if under_name else None,
            )

        if side_name:  # Check for equal rows
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

        if under_name:  # Check for equal columns
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

                if not value.columns.equals(under.columns):
                    raise ValueError(
                        f"{name} and {under_name} must have the same "
                        f"columns, got {value.columns} != {under.columns}."
                    )

        return value

    @property
    def pipeline(self) -> pd.Series:
        """Transformers fitted on the data.

        Use this attribute only to access the individual instances. To
        visualize the pipeline, use the [plot_pipeline][] method.

        """
        return self._pipeline

    @pipeline.setter
    @typechecked
    def pipeline(self, value: pd.Series):
        self._pipeline = value

    @property
    def mapping(self) -> CustomDict:
        """Encoded values and their respective mapped values.

        The column name is the key to its mapping dictionary. Only for
        columns mapped to a single column (e.g. Ordinal, Leave-one-out,
        etc...).

        """
        return self._mapping

    @mapping.setter
    @typechecked
    def mapping(self, value: CustomDict):
        self._mapping = value

    @property
    def dataset(self) -> pd.DataFrame:
        """Complete data set."""
        return self._data

    @dataset.setter
    @typechecked
    def dataset(self, value: X_TYPES):
        self._data = self._check_setter("dataset", value)

    @property
    def train(self) -> pd.DataFrame:
        """Training set."""
        return self._data.loc[self._idx[0], :]

    @train.setter
    @typechecked
    def train(self, value: X_TYPES):
        df = self._check_setter("train", value)
        self._data = self.T._set_index(pd.concat([df, self.test]))
        self._idx[0] = self._data.index[:len(df)]

    @property
    def test(self) -> pd.DataFrame:
        """Test set."""
        return self._data.loc[self._idx[1], :]

    @test.setter
    @typechecked
    def test(self, value: X_TYPES):
        df = self._check_setter("test", value)
        self._data = self.T._set_index(pd.concat([self.train, df]))
        self._idx[1] = self._data.index[-len(df):]

    @property
    def holdout(self) -> Optional[pd.DataFrame]:
        """Holdout set."""
        if self.T.holdout is not None and self._holdout is None:
            X, y = self.T.holdout.iloc[:, :-1], self.T.holdout.iloc[:, -1]
            for transformer in self.pipeline:
                if not transformer._train_only:
                    X, y = custom_transform(transformer, self, (X, y), verbose=0)

            self._holdout = merge(X, y)

        return self._holdout

    @property
    def X(self) -> pd.DataFrame:
        """Feature set."""
        return self._data.drop(self.target, axis=1)

    @X.setter
    @typechecked
    def X(self, value: X_TYPES):
        df = self._check_setter("X", value)
        self._data = merge(df, self.y)

    @property
    def y(self) -> pd.Series:
        """Target column."""
        return self._data[self.target]

    @y.setter
    @typechecked
    def y(self, value: SEQUENCE_TYPES):
        series = self._check_setter("y", value)
        self._data = merge(self._data.drop(self.target, axis=1), series)

    @property
    def X_train(self) -> pd.DataFrame:
        """Features of the training set."""
        return self.train.drop(self.target, axis=1)

    @X_train.setter
    @typechecked
    def X_train(self, value: X_TYPES):
        df = self._check_setter("X_train", value)
        self._data = pd.concat([merge(df, self.train[self.target]), self.test])

    @property
    def y_train(self) -> pd.Series:
        """Target column of the training set."""
        return self.train[self.target]

    @y_train.setter
    @typechecked
    def y_train(self, value: SEQUENCE_TYPES):
        series = self._check_setter("y_train", value)
        self._data = pd.concat([merge(self.X_train, series), self.test])

    @property
    def X_test(self) -> pd.DataFrame:
        """Features of the test set."""
        return self.test.drop(self.target, axis=1)

    @X_test.setter
    @typechecked
    def X_test(self, value: X_TYPES):
        df = self._check_setter("X_test", value)
        self._data = pd.concat([self.train, merge(df, self.test[self.target])])

    @property
    def y_test(self) -> pd.Series:
        """Target column of the test set."""
        return self.test[self.target]

    @y_test.setter
    @typechecked
    def y_test(self, value: SEQUENCE_TYPES):
        series = self._check_setter("y_test", value)
        self._data = pd.concat([self.train, merge(self.X_test, series)])

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the dataset (n_rows, n_cols)."""
        return self._data.shape

    @property
    def columns(self) -> pd.Series:
        """Name of all the columns."""
        return self._data.columns

    @property
    def n_columns(self) -> int:
        """Number of columns."""
        return len(self.columns)

    @property
    def features(self) -> pd.Series:
        """Name of the features."""
        return self.columns[:-1]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.features)

    @property
    def target(self) -> str:
        """Name of the target column."""
        return self.columns[-1]

    # Utility methods ============================================== >>

    def _get_attrs(self):
        """Get properties and attributes to call from parent.

        Returns
        -------
        list
            Properties and attributes.

        """
        attrs = []
        for p in dir(self):
            if (
                p in vars(self) and p not in ("T", "name", "_data", "idx", "_holdout")
                or isinstance(getattr(Branch, p, None), property)
            ):
                attrs.append(p)

        return attrs

    def _get_depending_models(self):
        """Return the models that are dependent on this branch.

        Returns
        -------
        list
            Depending models.

        """
        return [m.name for m in self.T._models.values() if m.branch is self]

    @composed(crash, method_to_log)
    def status(self):
        """Get an overview of the pipeline and models in the branch.

        This method prints the same information as the \__repr__ and
        also saves it to the logger.

        """
        self.T.log(str(self))

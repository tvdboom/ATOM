# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the Branch class.

"""

from __future__ import annotations

import re
from copy import copy
from functools import cached_property

import pandas as pd
from typeguard import typechecked

from atom.models import MODELS_ENSEMBLES
from atom.utils import (
    DATAFRAME, FEATURES, INDEX, INT, INT_TYPES, PANDAS, SEQUENCE, SERIES,
    SERIES_TYPES, TARGET, CustomDict, bk, custom_transform, flt, lst, merge,
    to_pandas,
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
    name: str
        Name of the branch.

    data: dataframe or None, default=None
        Complete dataset.

    index: list or None, default=None
        A list containing the number of target columns, the indices of
        the train set and the indices of the test set.

    holdout: dataframe or None, default=None
        Holdout dataset.

    parent: Branch or None, default=None
        Branch from which to split. If None, create an empty branch.

    """

    def __init__(
        self,
        name: str,
        data: DATAFRAME | None = None,
        index: list[INT, INDEX, INDEX] | None = None,
        holdout: DATAFRAME | None = None,
        parent: Branch | None = None,
    ):
        self._data = data
        self._idx = index
        self._holdout = holdout
        self._pipeline = pd.Series(data=[], dtype="object")
        self._mapping = CustomDict()

        # If a parent branch is provided, transfer its attrs to this one
        if parent:
            # Copy the data attrs (except holdout) and point to the rest
            for attr in ("_data", "_idx", "_pipeline", "_mapping"):
                setattr(self, attr, copy(getattr(parent, attr)))
            for attr in vars(parent):
                if not hasattr(self, attr):  # If not already assigned...
                    setattr(self, attr, getattr(parent, attr))

        self.name = name  # Name at end to change pipeline's name

    def __repr__(self) -> str:
        return f"Branch({self.name})"

    def __bool__(self):
        return self._data is not None

    @property
    def name(self) -> str:
        """Branch's name."""
        return self._name

    @name.setter
    @typechecked
    def name(self, value: str):
        if not value:
            raise ValueError("A branch can't have an empty name!")
        else:
            for model in MODELS_ENSEMBLES:
                if re.match(model.acronym, value, re.I):
                    raise ValueError(
                        "Invalid name for the branch. The name of a branch can "
                        f"not begin with a model's acronym, and {model.acronym} "
                        f"is the acronym of the {model._fullname} model."
                    )

        self._name = value
        self.pipeline.name = value

    # Data properties ============================================== >>

    def _check_setter(self, name: str, value: SEQUENCE | FEATURES) -> PANDAS:
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
        series or dataframe
            Data set.

        """

        def counter(name: str, dim: str) -> str:
            """Return the opposite dimension of the provided data set.

            Parameters
            ----------
            name: str
                Name of the data set.

            dim: str
                Dimension to look at. Either side or under.

            Returns
            -------
            str
                Name of the opposite dimension.

            """
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

        # Define the data attrs side and under
        if side_name := counter(name, "side"):
            side = getattr(self, side_name)
        if under_name := counter(name, "under"):
            under = getattr(self, under_name)

        value = to_pandas(
            data=value,
            index=side.index if side_name else None,
            name=getattr(under, "name", None) if under_name else None,
            columns=getattr(under, "columns", None) if under_name else None,
            dtype=under.dtypes if under_name else None,
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
            if isinstance(value, SERIES_TYPES):
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

    @property
    def mapping(self) -> CustomDict:
        """Encoded values and their respective mapped values.

        The column name is the key to its mapping dictionary. Only for
        columns mapped to a single column (e.g. Ordinal, Leave-one-out,
        etc...).

        """
        return self._mapping

    @property
    def dataset(self) -> DATAFRAME:
        """Complete data set."""
        return self._data

    @dataset.setter
    @typechecked
    def dataset(self, value: FEATURES):
        self._data = self._check_setter("dataset", value)

    @property
    def train(self) -> DATAFRAME:
        """Training set."""
        return self._data.loc[self._idx[1], :]

    @train.setter
    @typechecked
    def train(self, value: FEATURES):
        df = self._check_setter("train", value)
        self._data = bk.concat([df, self.test])
        self._idx[1] = self._data.index[:len(df)]

    @property
    def test(self) -> DATAFRAME:
        """Test set."""
        return self._data.loc[self._idx[2], :]

    @test.setter
    @typechecked
    def test(self, value: FEATURES):
        df = self._check_setter("test", value)
        self._data = bk.concat([self.train, df])
        self._idx[2] = self._data.index[-len(df):]

    @cached_property
    def holdout(self) -> DATAFRAME | None:
        """Holdout set."""
        if self._holdout is not None:
            X, y = self._holdout.iloc[:, :-self._idx[0]], self._holdout[self.target]
            for transformer in self.pipeline:
                if not transformer._train_only:
                    X, y = custom_transform(transformer, self, (X, y), verbose=0)

            return merge(X, y)

    @property
    def X(self) -> DATAFRAME:
        """Feature set."""
        return self._data.drop(self.target, axis=1)

    @X.setter
    @typechecked
    def X(self, value: FEATURES):
        df = self._check_setter("X", value)
        self._data = merge(df, self.y)

    @property
    def y(self) -> PANDAS:
        """Target column(s)."""
        return self._data[self.target]

    @y.setter
    @typechecked
    def y(self, value: TARGET):
        series = self._check_setter("y", value)
        self._data = merge(self._data.drop(self.target, axis=1), series)

    @property
    def X_train(self) -> DATAFRAME:
        """Features of the training set."""
        return self.train.drop(self.target, axis=1)

    @X_train.setter
    @typechecked
    def X_train(self, value: FEATURES):
        df = self._check_setter("X_train", value)
        self._data = bk.concat([merge(df, self.train[self.target]), self.test])

    @property
    def y_train(self) -> PANDAS:
        """Target column(s) of the training set."""
        return self.train[self.target]

    @y_train.setter
    @typechecked
    def y_train(self, value: TARGET):
        series = self._check_setter("y_train", value)
        self._data = bk.concat([merge(self.X_train, series), self.test])

    @property
    def X_test(self) -> DATAFRAME:
        """Features of the test set."""
        return self.test.drop(self.target, axis=1)

    @X_test.setter
    @typechecked
    def X_test(self, value: FEATURES):
        df = self._check_setter("X_test", value)
        self._data = bk.concat([self.train, merge(df, self.test[self.target])])

    @property
    def y_test(self) -> PANDAS:
        """Target column(s) of the test set."""
        return self.test[self.target]

    @y_test.setter
    @typechecked
    def y_test(self, value: TARGET):
        series = self._check_setter("y_test", value)
        self._data = bk.concat([self.train, merge(self.X_test, series)])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the dataset (n_rows, n_columns)."""
        return self._data.shape

    @property
    def columns(self) -> SERIES:
        """Name of all the columns."""
        return self._data.columns

    @property
    def n_columns(self) -> int:
        """Number of columns."""
        return len(self.columns)

    @property
    def features(self) -> SERIES:
        """Name of the features."""
        return self.columns[:-self._idx[0]]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.features)

    @property
    def target(self) -> str | list[str]:
        """Name of the target column(s)."""
        return flt(list(self.columns[-self._idx[0]:]))

    # Utility methods ============================================== >>

    def _get_rows(
        self,
        index: INT | str | slice | SEQUENCE | None,
        return_test: bool = True,
    ) -> list:
        """Get a subset of the rows in the dataset.

        Rows can be selected by name, index or regex pattern. If a
        string is provided, use `+` to select multiple rows and `!`
        to exclude them. Rows cannot be included and excluded in the
        same call.

        Parameters
        ----------
        index: int, str, slice, sequence or None, default=None
            Names or indices of the rows to select. If None, returns
            the complete dataset or the test set.

        return_test: bool, default=True
            Whether to return the test or the complete dataset when no
            index is provided.

        Returns
        -------
        list
            Indices of the included rows.

        """

        def get_match(idx: str, ex: ValueError | None = None):
            """Try to find a match by regex.

            Parameters
            ----------
            idx: str
                Regex pattern to match with indices.

            ex: ValueError or None
                Exception to raise if failed (from previous call).

            """
            nonlocal inc, exc

            array = inc
            if idx.startswith("!") and idx not in indices:
                array = exc
                idx = idx[1:]

            # Find rows using regex matches
            if matches := [i for i in indices if re.fullmatch(idx, str(i))]:
                array.extend(matches)
            else:
                raise ex or ValueError(
                    "Invalid value for the index parameter. "
                    f"Could not find any row that matches {idx}."
                )

        indices = list(self.dataset.index)
        # Note that this call caches the holdout calculation!
        if self.holdout is not None:
            indices += list(self.holdout.index)

        inc, exc = [], []
        if index is None:
            inc = list(self._idx[2]) if return_test else list(self.X.index)
        elif isinstance(index, slice):
            inc = indices[index]
        else:
            for idx in lst(index):
                if isinstance(idx, (*INT_TYPES, str)) and idx in indices:
                    inc.append(idx)
                elif isinstance(idx, INT_TYPES):
                    if -len(indices) <= idx <= len(indices):
                        inc.append(indices[idx])
                    else:
                        raise ValueError(
                            f"Invalid value for the index parameter. Value {index} is "
                            f"out of range for a dataset with {len(indices)} rows."
                        )
                elif isinstance(idx, str):
                    try:
                        get_match(idx)
                    except ValueError as ex:
                        for i in idx.split("+"):
                            get_match(i, ex)
                else:
                    raise TypeError(
                        f"Invalid type for the index parameter, got {type(idx)}. "
                        "Use a row's name or position to select it."
                    )

        if len(inc) + len(exc) == 0:
            raise ValueError(
                "Invalid value for the index parameter, got "
                f"{index}. At least one row has to be selected."
            )
        elif inc and exc:
            raise ValueError(
                "Invalid value for the index parameter. You can either "
                "include or exclude rows, not combinations of these."
            )

        if exc:
            # If rows were excluded with `!`, select all but those
            inc = [idx for idx in indices if idx not in exc]

        return list(dict.fromkeys(inc))  # Avoid duplicates

    def _get_columns(
        self,
        columns: INT | str | slice | SEQUENCE | None = None,
        include_target: bool = True,
        return_inc_exc: bool = False,
        only_numerical: bool = False,
    ) -> list[str] | tuple[list[str] | list[str]]:
        """Get a subset of the columns.

        Columns can be selected by name, index or regex pattern. If a
        string is provided, use `+` to select multiple columns and `!`
        to exclude them. Columns cannot be included and excluded in
        the same call.

        Parameters
        ----------
        columns: int, str, slice, sequence or None
            Names, indices or dtypes of the columns to select. If None,
            it returns all columns in the dataframe.

        include_target: bool, default=True
            Whether to include the target column in the dataframe to
            select from.

        return_inc_exc: bool, default=False
            Whether to return only included columns or the tuple
            (included, excluded).

        only_numerical: bool, default=False
            Whether to return only numerical columns.

        Returns
        -------
        list
            Names of the included columns.

        list
            Names of the excluded columns. Only returned if
            return_inc_exc=True.

        """

        def get_match(col: str, ex: ValueError | None = None):
            """Try to find a match by regex.

            Parameters
            ----------
            col: str
                Regex pattern to match with the column names.

            ex: ValueError or None
                Exception to raise if failed (from previous call).

            """
            nonlocal inc, exc

            array = inc
            if col.startswith("!") and col not in df.columns:
                array = exc
                col = col[1:]

            # Find columns using regex matches
            if matches := [c for c in df.columns if re.fullmatch(col, str(c))]:
                array.extend(matches)
            else:
                # Find columns by type
                try:
                    array.extend(list(df.select_dtypes(col).columns))
                except TypeError:
                    raise ex or ValueError(
                        "Invalid value for the columns parameter. "
                        f"Could not find any column that matches {col}."
                    )

        # Select dataframe from which to get the columns
        df = self.dataset if include_target else self.X

        inc, exc = [], []
        if columns is None:
            if only_numerical:
                return list(df.select_dtypes(include=["number"]).columns)
            else:
                return list(df.columns)
        elif isinstance(columns, slice):
            inc = list(df.columns[columns])
        else:
            for col in lst(columns):
                if isinstance(col, INT_TYPES):
                    try:
                        inc.append(df.columns[col])
                    except IndexError:
                        raise ValueError(
                            f"Invalid value for the columns parameter. Value {col} "
                            f"is out of range for a dataset with {df.shape[1]} columns."
                        )
                elif isinstance(col, str):
                    try:
                        get_match(col)
                    except ValueError as ex:
                        for c in col.split("+"):
                            get_match(c, ex)
                else:
                    raise TypeError(
                        f"Invalid type for the columns parameter, got {type(col)}. "
                        "Use a column's name or position to select it."
                    )

        if len(inc) + len(exc) == 0:
            raise ValueError(
                "Invalid value for the columns parameter, got "
                f"{columns}. At least one column has to be selected."
            )
        elif inc and exc:
            raise ValueError(
                "Invalid value for the columns parameter. You can either "
                "include or exclude columns, not combinations of these."
            )
        elif return_inc_exc:
            return list(dict.fromkeys(inc)), list(dict.fromkeys(exc))

        if exc:
            # If columns were excluded with `!`, select all but those
            inc = [col for col in df.columns if col not in exc]

        return list(dict.fromkeys(inc))  # Avoid duplicates

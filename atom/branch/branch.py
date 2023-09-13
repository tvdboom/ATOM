# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the Branch class.

"""

from __future__ import annotations

import os
import re
from functools import cached_property

import dill as pickle
from joblib.memory import Memory

from atom.pipeline import Pipeline
from atom.utils.types import (
    Bool, ColumnSelector, DataFrame, DataFrameTypes, Features, Index, Int,
    IntTypes, Pandas, RowSelector, Sequence, SeriesTypes, Target,
)
from atom.utils.utils import (
    CustomDict, DataContainer, bk, flt, get_cols, lst, merge, to_pandas,
)


class Branch:
    """Object that contains the data.

    A branch contains a specific pipeline, the dataset transformed
    through that pipeline, the models fitted on that dataset, and all
    data and utility attributes that refer to that dataset. Branches
    can be created and accessed through atom's `branch` attribute.

    All properties and attributes of the branch (except the private
    ones, starting with underscore) can be accessed from the parent.

    Read more in the [user guide][branches].

    Parameters
    ----------
    name: str
        Name of the branch.

    data: DataContainer or None, default=None
        Data for the branch.

    holdout: dataframe or None, default=None
        Holdout data set.

    See Also
    --------
    atom.branch:BranchManager

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

    def __init__(
        self,
        name: str,
        memory: Memory,
        data: DataContainer | None = None,
        holdout: DataFrame | None = None,
    ):
        self.name = name
        self.memory = memory

        self._container = data
        self._holdout = holdout
        self._pipeline = Pipeline([], memory=memory)
        self._mapping = CustomDict()

        # Path to store the data
        if memory.location is None:
            self._location = None
        else:
            self._location = os.path.join(memory.location, f"joblib/atom/{self}.pkl")

    def __repr__(self) -> str:
        return f"Branch({self.name})"

    @property
    def _data(self) -> DataFrame | None:
        """Get the branch's data.

        Load from memory if the data container is empty. This property
        is required to access the data from inactive branches.

        """
        return self.load(assign=False)

    @_data.setter
    def _data(self, value: DataContainer):
        self._container = value

    @property
    def name(self) -> str:
        """Branch's name."""
        return self._name

    @name.setter
    def name(self, value: str):
        from atom.models import MODELS_ENSEMBLES  # Avoid circular import

        if not value:
            raise ValueError("A branch can't have an empty name!")
        else:
            for model in MODELS_ENSEMBLES:
                if re.match(model.acronym, value, re.I):
                    raise ValueError(
                        "Invalid name for the branch. The name of a branch can "
                        f"not begin with a model's acronym, and {model.acronym} "
                        f"is the acronym of the {model.__name__} model."
                    )

        self._name = value

    # Data properties ============================================== >>

    def _check_setter(self, name: str, value: Sequence | Features) -> Pandas:
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
            name=getattr(under, "name", None) if under_name else "target",
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
            if isinstance(value, SeriesTypes):
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

        # Reset holdout calculation
        self.__dict__.pop("holdout", None)

        return value

    @property
    def pipeline(self) -> Pipeline:
        """Pipeline of transforms.

        !!! tip
            Use the [plot_pipeline][] method to visualize the pipeline.

        """
        return self._pipeline

    @property
    def mapping(self) -> CustomDict:
        """Encoded values and their respective mapped values.

        The column name is the key to its mapping dictionary. Only for
        columns mapped to a single column (e.g., Ordinal, Leave-one-out,
        etc...).

        """
        return self._mapping

    @property
    def dataset(self) -> DataFrame:
        """Complete data set."""
        return self._data.data

    @dataset.setter
    def dataset(self, value: Features):
        self._data.data = self._check_setter("dataset", value)

    @property
    def train(self) -> DataFrame:
        """Training set."""
        return self._data.data.loc[self._data.train_idx]

    @train.setter
    def train(self, value: Features):
        df = self._check_setter("train", value)
        self._data.data = bk.concat([df, self.test])
        self._data.train_idx = df.index

    @property
    def test(self) -> DataFrame:
        """Test set."""
        return self._data.data.loc[self._data.test_idx]

    @test.setter
    def test(self, value: Features):
        df = self._check_setter("test", value)
        self._data.data = bk.concat([self.train, df])
        self._data.test_idx = df.index

    @cached_property
    def holdout(self) -> DataFrame | None:
        """Holdout set."""
        if self._holdout is not None:
            return merge(
                *self.pipeline.transform(
                    X=self._holdout.iloc[:, :-self._data.n_cols],
                    y=self._holdout[self.target],
                    verbose=0,
                )
            )

    @property
    def X(self) -> DataFrame:
        """Feature set."""
        return self._data.data.drop(self.target, axis=1)

    @X.setter
    def X(self, value: Features):
        df = self._check_setter("X", value)
        self._data.data = merge(df, self.y)

    @property
    def y(self) -> Pandas:
        """Target column(s)."""
        return self._data.data[self.target]

    @y.setter
    def y(self, value: Target):
        series = self._check_setter("y", value)
        self._data.data = merge(self._data.data.drop(self.target, axis=1), series)

    @property
    def X_train(self) -> DataFrame:
        """Features of the training set."""
        return self.train.drop(self.target, axis=1)

    @X_train.setter
    def X_train(self, value: Features):
        df = self._check_setter("X_train", value)
        self._data.data = bk.concat([merge(df, self.train[self.target]), self.test])

    @property
    def y_train(self) -> Pandas:
        """Target column(s) of the training set."""
        return self.train[self.target]

    @y_train.setter
    def y_train(self, value: Target):
        series = self._check_setter("y_train", value)
        self._data.data = bk.concat([merge(self.X_train, series), self.test])

    @property
    def X_test(self) -> DataFrame:
        """Features of the test set."""
        return self.test.drop(self.target, axis=1)

    @X_test.setter
    def X_test(self, value: Features):
        df = self._check_setter("X_test", value)
        self._data.data = bk.concat([self.train, merge(df, self.test[self.target])])

    @property
    def y_test(self) -> Pandas:
        """Target column(s) of the test set."""
        return self.test[self.target]

    @y_test.setter
    def y_test(self, value: Target):
        series = self._check_setter("y_test", value)
        self._data.data = bk.concat([self.train, merge(self.X_test, series)])

    @property
    def shape(self) -> tuple[Int, Int]:
        """Shape of the dataset (n_rows, n_columns)."""
        return self._data.data.shape

    @property
    def columns(self) -> Index:
        """Name of all the columns."""
        return self._data.data.columns

    @property
    def n_columns(self) -> Int:
        """Number of columns."""
        return len(self.columns)

    @property
    def features(self) -> Index:
        """Name of the features."""
        return self.columns[:-self._data.n_cols]

    @property
    def n_features(self) -> Int:
        """Number of features."""
        return len(self.features)

    @property
    def target(self) -> str | list[str]:
        """Name of the target column(s)."""
        return flt(list(self.columns[-self._data.n_cols:]))

    @property
    def _all(self) -> DataFrame:
        """Dataset + holdout.

        Note that calling this property triggers the holdout set
        calculation.

        """
        return bk.concat([self.dataset, self.holdout])

    # Utility methods ============================================== >>

    def _get_rows(self, rows: RowSelector) -> DataFrame:
        """Get a subset of the data.

        Rows can be selected by name, index, regex pattern or data set.
        If a string is provided, use `+` to select multiple rows and `!`
        to exclude them. Rows cannot be included and excluded in the
        same call.

        !!! note
            This call activates the holdout calculation for this branch.

        Parameters
        ----------
        rows: hashable, range, slice or sequence
            Rows to select.

        Returns
        -------
        dataframe
            Subset of the data.

        """
        indices = self._all.index

        inc, exc = [], []
        if isinstance(rows, (range, slice)):
            return self._all.loc[indices[rows]]
        else:
            for row in lst(rows):
                if row in indices:
                    inc.append(row)
                elif isinstance(row, IntTypes):
                    if -len(indices) <= row <= len(indices):
                        inc.append(indices[row])
                    else:
                        raise IndexError(
                            f"Invalid value for the rows parameter. Value {rows} "
                            f"is out of range for data with {len(indices)} rows."
                        )
                elif isinstance(row, str):
                    for r in row.split("+"):
                        array = inc
                        if r.startswith("!") and r not in indices:
                            array = exc
                            r = r[1:]

                        # Find match on data set
                        if r.lower() in ("dataset", "train", "test", "holdout"):
                            try:
                                array.extend(getattr(self, r.lower()).index)
                            except AttributeError:
                                raise ValueError(
                                    "Invalid value for the rows parameter. No holdout "
                                    "data set was declared when initializing atom."
                                )
                        elif (matches := indices.str.fullmatch(r)).sum() > 0:
                            array.extend(indices[matches])

        if len(inc) + len(exc) == 0:
            raise ValueError(
                "Invalid value for the rows parameter, got "
                f"{rows}. No rows were selected."
            )
        elif inc and exc:
            raise ValueError(
                "Invalid value for the rows parameter. You can either "
                "include or exclude rows, not combinations of these."
            )

        if exc:
            # If rows were excluded with `!`, select all but those
            inc = indices[~indices.isin(exc)]

        return self._all.loc[inc]

    def _get_columns(
        self,
        columns: ColumnSelector | None = None,
        include_target: Bool = True,
        only_numerical: Bool = False,
    ) -> list[str]:
        """Get a subset of the columns.

        Columns can be selected by name, index or regex pattern. If a
        string is provided, use `+` to select multiple columns and `!`
        to exclude them. Columns cannot be included and excluded in
        the same call.

        Parameters
        ----------
        columns: int, str, range, slice, sequence or None
            Names, indices or dtypes of the columns to select. If None,
            it returns all columns in the dataframe.

        include_target: bool, default=True
            Whether to include the target column in the dataframe to
            select from.

        only_numerical: bool, default=False
            Whether to return only numerical columns.

        Returns
        -------
        list of str
            Names of the included columns.

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
                if isinstance(col, IntTypes):
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

        if inc and exc:
            raise ValueError(
                "Invalid value for the columns parameter. You can either "
                "include or exclude columns, not combinations of these."
            )
        elif exc:
            # If columns were excluded with `!`, select all but those
            inc = df.columns[~df.columns.isin(exc)]

        if len(inc) == 0:
            raise ValueError(
                "Invalid value for the columns parameter, got "
                f"{columns}. At least one column has to be selected."
            )

        return list(dict.fromkeys(inc))  # Avoid duplicates

    def _get_target(
        self,
        target: Int | str | tuple,
        only_columns: Bool = False,
    ) -> str | tuple[Int, Int]:
        """Get a target column and/or class in target column.

        Parameters
        ----------
        target: int, str or tuple
            Target column or class to retrieve. For multioutput tasks,
            use a tuple of the form (column, class) to select a class
            in a specific target column.

        only_columns: bool, default=False
            Whether to only look at target columns or also to target
            classes (for multilabel and multiclass-multioutput tasks).

        Returns
        -------
        str or tuple
            Name of the selected target column (if only_columns=True)
            or tuple of the form (column, class).

        """

        def get_column(target: Int | str) -> str:
            """Get the target column.

            Parameters
            ----------
            target: int or str
                Name or position of the target column.

            Returns
            -------
            str
                Target column.

            """
            if isinstance(target, str):
                if target not in self.target:
                    raise ValueError(
                        "Invalid value for the target parameter. Value "
                        f"{target} is not one of the target columns."
                    )
                else:
                    return target
            else:
                if not 0 <= target < len(self.target):
                    raise ValueError(
                        "Invalid value for the target parameter. There are "
                        f"{len(self.target)} target columns, got {target}."
                    )
                else:
                    return lst(self.target)[target]

        def get_class(target: Int | str, column: int = 0) -> int:
            """Get the class in the target column.

            Parameters
            ----------
            target: int or str
                Name or position of the target column.

            column: int, default=0
                Column to get the class from. For multioutput tasks.

            Returns
            -------
            int
                Class' index.

            """
            if isinstance(target, str):
                try:
                    return self.mapping[lst(self.target)[column]][target]
                except (TypeError, KeyError):
                    raise ValueError(
                        f"Invalid value for the target parameter. Value {target} "
                        "not found in the mapping of the target column."
                    )
            else:
                n_classes = get_cols(self.y)[column].nunique(dropna=False)
                if not 0 <= target < n_classes:
                    raise ValueError(
                        "Invalid value for the target parameter. "
                        f"There are {n_classes} classes, got {target}."
                    )
                else:
                    return target

        if only_columns:
            return get_column(target)
        elif isinstance(target, tuple):
            if not isinstance(self.y, DataFrameTypes):
                raise ValueError(
                    f"Invalid value for the target parameter, got {target}. "
                    "A tuple is only accepted for multioutput tasks."
                )
            elif len(target) == 1:
                return self.target.index(get_column(target[0])), 0
            elif len(target) == 2:
                column = self.target.index(get_column(target[0]))
                return column, get_class(target[1], column)
            else:
                raise ValueError(
                    "Invalid value for the target parameter. "
                    f"Expected a tuple of length 2, got len={len(target)}."
                )
        else:
            return 0, get_class(target)

    def load(self, assign: Bool = True) -> DataContainer | None:
        """Load the branch's data from memory.

        This method is used to restore the data of inactive branches.

        Parameters
        ----------
        assign: bool, default=True
            Whether to assign the loaded data to `self`.

        Returns
        -------
        DataContainer or None
            Own data information. Returns None if no data is set.

        """
        if self._container is None and self._location:
            try:
                with open(self._location, "rb") as file:
                    data = pickle.load(file)
            except FileNotFoundError:
                raise ValueError(f"Branch {self.name} has no data.")

            if assign:
                self._container = data
            else:
                return data

        return self._container

    def store(self):
        """Store the branch's data as a pickle in memory.

        After storage, the data is deleted and the branch is no longer
        usable until [load][self-load] is called. This method is used
        to store the data for inactive branches.

        !!! note
            This method is skipped silently for branches with no memory
            allocation.

        """
        if self._location:
            with open(self._location, "wb") as file:
                pickle.dump(self._container, file)

            self._container = None

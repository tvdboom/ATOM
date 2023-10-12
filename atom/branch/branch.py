# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the Branch class.

"""

from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path
from typing import overload

import dill as pickle
from beartype import beartype
from beartype.typing import Hashable, Literal
from joblib.memory import Memory
from sklearn.utils.validation import check_memory

from atom.pipeline import Pipeline
from atom.utils.types import (
    Bool, ColumnSelector, DataFrame, DataFrameTypes, Features, Index, Int,
    IntLargerEqualZero, IntTypes, Pandas, RowSelector, Scalar, SegmentTypes,
    Sequence, SeriesTypes, Target, TargetSelector, TargetsSelector,
)
from atom.utils.utils import (
    CustomDict, DataContainer, bk, flt, get_cols, lst, merge, to_pandas,
)


@beartype
class Branch:
    """Object that contains the data.

    A branch contains a specific pipeline, the dataset transformed
    through that pipeline, the models fitted on that dataset, and all
    data and utility attributes that refer to that dataset. Branches
    can be created and accessed through atom's `branch` attribute.

    All public properties and attributes of the branch can be accessed
    from the parent.

    Read more in the [user guide][branches].

    !!! warning
        This class should not be called directly. Branches are created
        internally by the [ATOMClassifier][], [ATOMForecaster][] and
        [ATOMRegressor][] classes.

    Parameters
    ----------
    name: str
        Name of the branch.

    memory: str, [Memory][joblibmemory] or None, default=None
        Memory object for pipeline caching and to store the data when
        the branch is inactive.

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
        memory: str | Memory | None = None,
        data: DataContainer | None = None,
        holdout: DataFrame | None = None,
    ):
        self.name = name
        self.memory = check_memory(memory)

        self._container = data
        self._holdout = holdout
        self._pipeline = Pipeline([], memory=memory)
        self._mapping = CustomDict()

        # Path to store the data
        if self.memory.location is None:
            self._location = None
        else:
            self._location = Path(self.memory.location).joinpath("joblib", "atom")

    def __repr__(self) -> str:
        return f"Branch({self.name})"

    @property
    def _data(self) -> DataContainer:
        """Get the branch's data.

        Load from memory if the data container is empty. This property
        is required to access the data for inactive branches.

        """
        if data := self.load(assign=False):
            return data

        raise RuntimeError("This branch has no dataset assigned.")

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

        self._name: str = value

    # Data properties ============================================== >>

    def _check_setter(
        self,
        name: str,
        value: Sequence[Scalar | str] | Features,
    ) -> Pandas:
        """Check the data set's setter property.

        Convert the property to a pandas object and compare with the
        rest of the dataset, to check if it has the right indices and
        dimensions.

        Parameters
        ----------
        name: str
            Name of the data set to check.

        value: sequence or dataframe-like
            New values for the data set.

        Returns
        -------
        series or dataframe
            Data set.

        """

        def counter(name: str, dim: str) -> str | None:
            """Return the opposite dimension of the provided data set.

            Parameters
            ----------
            name: str
                Name of the data set.

            dim: str
                Dimension to look at. Either side or under.

            Returns
            -------
            str or None
                Name of the opposite dimension. Returns None when there
                is no opposite dimension, e.g., train with dim="side".

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

            return None

        # Define the data attrs side and under
        if side_name := counter(name, "side"):
            side = getattr(self, side_name)
        if under_name := counter(name, "under"):
            under = getattr(self, under_name)

        obj: Pandas = to_pandas(
            data=value,
            index=side.index if side_name else None,
            name=getattr(under, "name", "target") if under_name else "target",
            columns=getattr(under, "columns", None) if under_name else None,
            dtype=under.dtypes if under_name else None,
        )

        if side_name:  # Check for equal rows
            if len(obj) != len(side):
                raise ValueError(
                    f"{name} and {side_name} must have the same "
                    f"number of rows, got {len(obj)} != {len(side)}."
                )
            if not obj.index.equals(side.index):
                raise ValueError(
                    f"{name} and {side_name} must have the same "
                    f"indices, got {obj.index} != {side.index}."
                )

        if under_name:  # Check for equal columns
            if isinstance(obj, SeriesTypes):
                if obj.name != under.name:
                    raise ValueError(
                        f"{name} and {under_name} must have the "
                        f"same name, got {obj.name} != {under.name}."
                    )
            else:
                if obj.shape[1] != under.shape[1]:
                    raise ValueError(
                        f"{name} and {under_name} must have the same number "
                        f"of columns, got {obj.shape[1]} != {under.shape[1]}."
                    )

                if not obj.columns.equals(under.columns):
                    raise ValueError(
                        f"{name} and {under_name} must have the same "
                        f"columns, got {obj.columns} != {under.columns}."
                    )

        # Reset holdout calculation
        self.__dict__.pop("holdout", None)

        return obj

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
                )
            )
        else:
            return None

    @property
    def X(self) -> DataFrame:
        """Feature set."""
        return self._data.data[self.features]

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
        self._data.data = merge(self.X, series)

    @property
    def X_train(self) -> DataFrame:
        """Features of the training set."""
        return self.train[self.features]

    @X_train.setter
    def X_train(self, value: Features):
        df = self._check_setter("X_train", value)
        self._data.data = bk.concat([merge(df, self.y_train), self.test])

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
        return self.test[self.features]

    @X_test.setter
    def X_test(self, value: Features):
        df = self._check_setter("X_test", value)
        self._data.data = bk.concat([self.train, merge(df, self.y_test)])

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
        return self.dataset.shape

    @property
    def columns(self) -> Index:
        """Name of all the columns."""
        return self.dataset.columns

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

    @overload
    def _get_rows(
        self,
        rows: RowSelector,
        return_X_y: Literal[False] = ...,
    ) -> DataFrame: ...

    @overload
    def _get_rows(
        self,
        rows: RowSelector,
        return_X_y: Literal[True],
    ) -> tuple[DataFrame, Pandas]: ...

    def _get_rows(
        self,
        rows: RowSelector,
        return_X_y: Bool = False,
    ) -> DataFrame | tuple[DataFrame, Pandas]:
        """Get a subset of the rows.

        Rows can be selected by name, index, data set or regex pattern.
        If a string is provided, use `+` to select multiple rows and `!`
        to exclude them. Rows cannot be included and excluded in the
        same call.

        !!! note
            This call activates the holdout calculation for this branch.

        Parameters
        ----------
        rows: hashable, segment, sequence or dataframe
            Rows to select.

        return_X_y: bool, default=False
            Whether to return X and y separately or concatenated.

        Returns
        -------
        dataframe
            Subset of rows.

        series or dataframe
            Subset of target column. Only returned if return_X_y=True.

        """
        indices = self._all.index

        inc: list[Hashable] = []
        exc: list[Hashable] = []
        if isinstance(rows, DataFrameTypes):
            inc.extend(rows.index)
        elif isinstance(rows, SegmentTypes):
            inc.extend(indices[rows])
        else:
            for row in lst(rows):
                if row in indices:
                    inc.append(row)
                elif isinstance(row, IntTypes):
                    if -len(indices) <= row < len(indices):
                        inc.append(indices[int(row)])
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
                                ) from None
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
        elif exc:
            # If rows were excluded with `!`, select all but those
            inc = list(indices[~indices.isin(exc)])

        if return_X_y:
            X, y = self._all[self.features], self._all[self.target]
            return X.loc[inc], y.loc[inc]
        else:
            return self._all.loc[inc]

    def _get_columns(
        self,
        columns: ColumnSelector | None = None,
        include_target: Bool = True,
        only_numerical: Bool = False,
    ) -> list[str]:
        """Get a subset of the columns.

        Columns can be selected by name, index, dtype or regex pattern.
        If a string is provided, use `+` to select multiple columns and
        `!` to exclude them. Columns cannot be included and excluded in
        the same call.

        Parameters
        ----------
        columns: ColumnSelector or None, default=None
            Columns to select. If None, return all columns in the
            dataset, bearing the other parameters.

        include_target: bool, default=True
            Whether to include the target column in the dataset to
            select from.

        only_numerical: bool, default=False
            Whether to select only numerical columns.

        Returns
        -------
        list of str
            Names of the included columns.

        """
        # Select dataframe from which to get the columns
        df = self.dataset if include_target else self.X

        inc: list[str] = []
        exc: list[str] = []
        if columns is None:
            if only_numerical:
                return list(df.select_dtypes(include=["number"]).columns)
            else:
                return list(df.columns)
        elif isinstance(columns, DataFrameTypes):
            inc.extend(list(columns.columns))
        elif isinstance(columns, SegmentTypes):
            inc.extend(list(df.columns[columns]))
        else:
            for col in lst(columns):
                if isinstance(col, IntTypes):
                    if -df.shape[1] <= col < df.shape[1]:
                        inc.append(df.columns[int(col)])
                    else:
                        raise IndexError(
                            f"Invalid value for the columns parameter. Value {col} "
                            f"is out of range for data with {df.shape[1]} columns."
                        )
                elif isinstance(col, str):
                    for c in col.split("+"):
                        array = inc
                        if c.startswith("!") and c not in df.columns:
                            array = exc
                            c = c[1:]

                        # Find columns using regex matches
                        if (matches := df.columns.str.fullmatch(c)).sum() > 0:
                            array.extend(df.columns[matches])
                        else:
                            # Find columns by type
                            try:
                                array.extend(df.select_dtypes(c).columns)
                            except TypeError:
                                raise ValueError(
                                    "Invalid value for the columns parameter. "
                                    f"Could not find any column that matches {c}."
                                ) from None

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
        elif exc:
            # If columns were excluded with `!`, select all but those
            inc = list(df.columns[~df.columns.isin(exc)])

        return list(dict.fromkeys(inc))  # Avoid duplicates

    @overload
    def _get_target(
        self,
        target: TargetsSelector,
        only_columns: Literal[False] = ...,
    ) -> tuple[IntLargerEqualZero, IntLargerEqualZero]: ...

    @overload
    def _get_target(
        self,
        target: TargetsSelector,
        only_columns: Literal[True],
    ) -> str: ...

    def _get_target(
        self,
        target: TargetsSelector,
        only_columns: Bool = False,
    ) -> str | tuple[IntLargerEqualZero, IntLargerEqualZero]:
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

        def get_column(target: TargetSelector) -> str:
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

        def get_class(
            target: TargetSelector,
            column: IntLargerEqualZero = 0,
        ) -> IntLargerEqualZero:
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

        if only_columns and not isinstance(target, tuple):
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
                with open(self._location.joinpath(f"{self}.pkl"), "rb") as file:
                    data = pickle.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Branch {self.name} has no data stored.")

            if assign:
                self._container = data
            else:
                return data

        return self._container

    def store(self, assign: Bool = True):
        """Store the branch's data as a pickle in memory.

        After storage, the data is deleted, and the branch is no longer
        usable until [load][self-load] is called. This method is used
        to store the data for inactive branches.

        !!! note
            This method is skipped silently for branches with no memory
            allocation.

        Parameters
        ----------
        assign: bool, default=True
            Whether to assign `None` to the data in `self`.

        """
        if self._container is not None and self._location:
            try:
                with open(self._location.joinpath(f"{self}.pkl"), "wb") as file:
                    pickle.dump(self._container, file)
            except FileNotFoundError:
                raise FileNotFoundError(f"The {self._location} directory does not exist.")

            if assign:
                self._container = None

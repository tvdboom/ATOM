"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the BaseRunner class.

"""

from __future__ import annotations

import math
import random
import re
from abc import ABCMeta
from collections.abc import Hashable
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Any

import dill as pickle
import numpy as np
import pandas as pd
from beartype import beartype
from pandas.tseries.frequencies import to_offset
from pmdarima.arima.utils import ndiffs
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.metaestimators import available_if
from sktime.datatypes import check_is_mtype
from sktime.param_est.seasonality import SeasonalityACF
from sktime.transformations.series.difference import Differencer

from atom.basetracker import BaseTracker
from atom.basetransformer import BaseTransformer
from atom.branch import Branch
from atom.models import MODELS, Stacking, Voting
from atom.pipeline import Pipeline
from atom.utils.constants import DF_ATTRS
from atom.utils.types import (
    Bool, FloatZeroToOneExc, HarmonicsSelector, IndexSelector, Int,
    IntLargerOne, MetricConstructor, Model, ModelSelector, ModelsSelector,
    Pandas, RowSelector, Seasonality, Segment, Sequence, SPDict, SPTuple,
    TargetSelector, YSelector, bool_t, int_t, pandas_t, segment_t, sequence_t,
)
from atom.utils.utils import (
    ClassMap, DataContainer, Goal, SeasonalPeriod, Task, check_is_fitted,
    composed, crash, divide, flt, get_cols, get_segment, get_versions,
    has_task, lst, merge, method_to_log, n_cols,
)


class BaseRunner(BaseTracker, metaclass=ABCMeta):
    """Base class for runners.

    Contains shared attributes and methods for atom and trainers.

    """

    def __getstate__(self) -> dict[str, Any]:
        """Require storing an extra attribute with the package versions."""
        return self.__dict__ | {"_versions": get_versions(self._models)}

    def __setstate__(self, state: dict[str, Any]):
        """Check if the loaded versions match with those installed."""
        versions = state.pop("_versions", None)
        self.__dict__.update(state)

        # Check that all package versions match or raise a warning
        if versions:
            for key, value in get_versions(state["_models"]).items():
                if versions[key] != value:
                    self._log(
                        f"The loaded instance used the {key} package with version "
                        f"{versions[key]} while the version in this environment is "
                        f"{value}.",
                        1,
                        severity="warning",
                    )

    def __dir__(self) -> list[str]:
        """Add additional attrs from __getattr__ to the dir."""
        # Exclude from _available_if conditions
        attrs = [x for x in super().__dir__() if hasattr(self, x)]

        # Add additional attrs from the branch
        attrs += self.branch._get_shared_attrs()

        # Add additional attrs from the dataset
        attrs += [x for x in DF_ATTRS if hasattr(self.dataset, x)]

        # Add branch names in lower-case
        attrs += [b.name.lower() for b in self._branches]

        # Add column names (excluding those with spaces)
        attrs += [c for c in self.columns if re.fullmatch(r"\w+$", c)]

        # Add model names in lower-case
        if isinstance(self._models, ClassMap):
            attrs += [m.name.lower() for m in self._models]

        return attrs

    def __getattr__(self, item: str) -> Any:
        """Get branch, attr from branch, model, column or attr from dataset."""
        if item in self.__dict__["_branches"]:
            return self._branches[item]  # Get branch
        elif item in self.branch._get_shared_attrs():
            if isinstance(attr := getattr(self.branch, item), pandas_t):
                return self._convert(attr)  # Transform data through data engine
            else:
                return attr
        elif item in self.__dict__["_models"]:
            return self._models[item]  # Get model
        elif item in self.branch.columns:
            return self.branch.dataset[item]  # Get column from dataset
        elif item in DF_ATTRS and hasattr(self.dataset, item):
            return getattr(self.dataset, item)  # Get attr from dataset
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'.")

    def __setattr__(self, item: str, value: Any):
        """Set attr to branch when it's a property of Branch."""
        if isinstance(getattr(Branch, item, None), property):
            setattr(self.branch, item, value)
        else:
            super().__setattr__(item, value)

    def __delattr__(self, item: str):
        """Delete model."""
        if item in self._models:
            self.delete(item)
        else:
            super().__delattr__(item)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.branch.dataset)

    def __contains__(self, item: str) -> bool:
        """Whether the item is a column in the dataset."""
        return item in self.dataset

    def __getitem__(self, item: Int | str | list) -> Any:
        """Get a branch, model or column from the dataset."""
        if self.branch._container is None:
            raise RuntimeError(
                "This instance has no dataset annexed to it. "
                "Use the run method before calling __getitem__."
            )
        elif isinstance(item, int_t):
            return self.dataset[self.columns[item]]
        elif isinstance(item, str):
            if item in self._branches:
                return self._branches[item]  # Get branch
            elif item in self._models:
                return self._models[item]  # Get model
            elif item in self.dataset:
                return self.dataset[item]  # Get column from dataset
            else:
                raise ValueError(
                    f"{self.__class__.__name__} object has no "
                    f"branch, model or column called {item}."
                )
        else:
            return self.dataset[item]  # Get subset of dataset

    def __sklearn_is_fitted__(self) -> bool:
        """Return fitted when there are trained models."""
        return bool(self._models)

    # Utility properties =========================================== >>

    @cached_property
    def task(self) -> Task:
        """Dataset's [task][] type."""
        return self._goal.infer_task(self.branch.y)

    @property
    def sp(self) -> SPTuple:
        """Seasonality of the time series.

        Read more about seasonality in the [user guide][seasonality].

        """
        return self._config.sp

    @sp.setter
    @beartype
    def sp(self, sp: Seasonality | SPDict):
        """Convert seasonality to information container."""
        if sp is None:
            self._config.sp = SPTuple()
        elif isinstance(sp, dict):
            self._config.sp = SPTuple(
                sp=self._get_sp(sp.get("sp", SPTuple().sp)),
                seasonal_model=sp.get("seasonal_model", SPTuple().seasonal_model),
                trend_model=sp.get("trend_model", SPTuple().trend_model),
            )
        else:
            self._config.sp = SPTuple(sp=self._get_sp(sp))

    @property
    def og(self) -> Branch:
        """Branch containing the original dataset.

        This branch contains the data prior to any transformations.
        It redirects to the current branch if its pipeline is empty
        to not have the same data in memory twice.

        """
        return self._branches.og

    @property
    def branch(self) -> Branch:
        """Current active branch."""
        return self._branches.current

    @property
    def holdout(self) -> pd.DataFrame | None:
        """Holdout set.

        This data set is untransformed by the pipeline. Read more in
        the [user guide][data-sets].

        """
        return self._convert(self.branch._holdout)

    @property
    def models(self) -> str | list[str] | None:
        """Name of the model(s)."""
        if self._models:
            return flt(self._models.keys())
        else:
            return None

    @property
    def metric(self) -> str | list[str] | None:
        """Name of the metric(s)."""
        if self._metric:
            return flt(self._metric.keys())
        else:
            return None

    @property
    def winners(self) -> list[Model] | None:
        """Models ordered by performance.

        Performance is measured as the highest score on the model's
        `[main_metric]_bootstrap` or `[main_metric]_test`, checked in
        that order. Ties are resolved looking at the lowest `time_fit`.

        """
        if self._models:  # Returns None if not fitted
            return sorted(self._models, key=lambda x: (x._best_score(), x._time_fit), reverse=True)
        else:
            return None

    @property
    def winner(self) -> Model | None:
        """Best performing model.

        Performance is measured as the highest score on the model's
        `[main_metric]_bootstrap` or `[main_metric]_test`, checked in
        that order. Ties are resolved looking at the lowest `time_fit`.

        """
        if self.winners:  # Returns None if not fitted
            return self.winners[0]
        else:
            return None

    @winner.deleter
    def winner(self):
        """[Delete][atomclassifier-delete] the best performing model."""
        if self._models:  # Do nothing if not fitted
            self.delete(self.winner.name)

    @property
    def results(self) -> pd.DataFrame:
        """Overview of the training results.

        All durations are in seconds. Possible values include:

        - **[metric]_ht:** Score obtained by the hyperparameter tuning.
        - **time_ht:** Duration of the hyperparameter tuning.
        - **[metric]_train:** Metric score on the train set.
        - **[metric]_test:** Metric score on the test set.
        - **time_fit:** Duration of the model fitting on the train set.
        - **[metric]_bootstrap:** Mean score on the bootstrapped samples.
        - **time_bootstrap:** Duration of the bootstrapping.
        - **time:** Total duration of the run.

        """

        def frac(m: Model) -> float:
            """Return the fraction of the train set used.

            Parameters
            ----------
            m: Model
                Model used.

            Returns
            -------
            float
                Calculated fraction.

            """
            if (n_models := len(m.branch.train) / m._train_idx) == int(n_models):
                return round(1.0 / n_models, 2)
            else:
                return round(m._train_idx / len(m.branch.train), 2)

        df = pd.DataFrame(
            data=[m.results for m in self._models],
            columns=self._models[0].results.index if self._models else [],
            index=lst(self.models),
        ).dropna(axis=1, how="all")

        # For sh and ts runs, include the fraction of training set
        if any(m._train_idx != len(m.branch.train) for m in self._models):
            df = df.set_index(
                pd.MultiIndex.from_arrays(
                    arrays=[[frac(m) for m in self._models], self.models],
                    names=["frac", "model"],
                )
            ).sort_index(level=0, ascending=True)

        return df

    # Utility methods ============================================== >>

    def _get_sp(self, sp: Seasonality) -> int | list[int] | None:
        """Get the seasonal period.

        Parameters
        ----------
        sp: int, str, sequence or None
            Seasonal period provided. If None, it means there is
            no seasonality.

        Returns
        -------
        int, list or None
            Seasonal period.

        """

        def get_single_sp(sp: Int | str) -> int:
            """Get a seasonal period from a single value.

            Parameters
            ----------
            sp: int, str or None
                Seasonal period as int or [DateOffset][].

            Returns
            -------
            int
                Seasonal period.

            """
            if isinstance(sp, str):
                if offset := to_offset(sp):  # Convert to pandas' DateOffset
                    name, period = offset.name.split("-")[0], offset.n

                if name not in SeasonalPeriod.__members__:
                    raise ValueError(
                        f"Invalid value for the seasonal period, got {name}. "
                        f"Valid values are: {', '.join(SeasonalPeriod.__members__)}"
                    )

                # Formula is same as SeasonalPeriod[name] for period=1
                return math.lcm(SeasonalPeriod[name].value, period) // period
            else:
                return int(sp)

        if sp is None:
            return None
        elif sp == "infer":
            return self.get_seasonal_period()
        elif sp == "index":
            if not hasattr(self.dataset.index, "freqstr"):
                raise ValueError(
                    f"Invalid value for the seasonal period, got {sp}. "
                    f"The dataset's index has no attribute freqstr."
                )
            else:
                return get_single_sp(self.dataset.index.freqstr)
        else:
            return flt([get_single_sp(x) for x in lst(sp)])

    def _get_data(
        self,
        arrays: tuple[Any, ...],
        y: YSelector = -1,
        *,
        index: IndexSelector | None = None,
    ) -> tuple[DataContainer, pd.DataFrame | None]:
        """Get data sets from a sequence of indexables.

        Also assigns an index, (stratified) shuffles and selects a
        subsample of rows depending on the attributes.

        Parameters
        ----------
        arrays: tuple of indexables
            Data set(s) provided. Should follow the API input format.

        y: int, str or sequence, default=-1
            Transformed target column.

        index: bool, int, str, sequence or None, default=None
            Index parameter as provided in constructor. If None, the
            index is retrieved from `self._config`.

        Returns
        -------
        DataContainer
            Train and test sets.

        pd.DataFrame or None
            Holdout data set. Returns None if not specified.

        """

        def _subsample(df: pd.DataFrame) -> pd.DataFrame:
            """Select a random subset of a dataframe.

            If shuffle=True, the subset is shuffled, else row order
            is maintained. For forecasting tasks, rows are dropped
            from the tail of the data set.

            Parameters
            ----------
            df: pd.DataFrame
                Dataset.

            Returns
            -------
            pd.DataFrame
                Subset of df.

            """
            if self._config.n_rows <= 1:
                n_rows = int(len(df) * self._config.n_rows)
            else:
                n_rows = int(self._config.n_rows)

            if self._config.shuffle:
                return df.iloc[random.sample(range(len(df)), k=n_rows)]
            elif self._goal.name == "forecast":
                return df.iloc[-n_rows:]  # For forecasting, select from tail
            else:
                return df.iloc[sorted(random.sample(range(len(df)), k=n_rows))]

        def _set_index(
            df: DataFrame,
            y: Pandas | None,
            index: IndexSelector | None = None,
        ) -> pd.DataFrame:
            """Assign an index to the dataframe.

            Parameters
            ----------
            df: pd.DataFrame
                Dataset.

            y: pd.Series, pd.DataFrame or None
                Target column(s). Used to check that the provided index
                is not one of the target columns. If None, the check is
                skipped.

            index: bool, int, str or sequence or None, default=None
                Index parameter as provided in constructor. If None, the
                index is retrieved from `self._config`.

            Returns
            -------
            pd.DataFrame
                Dataset with updated indices.

            """
            if index is None:
                index = self._config.index

            if index is True:  # True gets caught by isinstance(int)
                pass
            elif index is False:
                df = df.reset_index(drop=True)
            elif isinstance(index, int_t):
                if -df.shape[1] <= index <= df.shape[1]:
                    df = df.set_index(df.columns[int(index)], drop=True)
                else:
                    raise IndexError(
                        f"Invalid value for the index parameter. Value {index} "
                        f"is out of range for a dataset with {df.shape[1]} columns."
                    )
            elif isinstance(index, str):
                if index in df:
                    df = df.set_index(index, drop=True)
                else:
                    raise ValueError(
                        "Invalid value for the index parameter. "
                        f"Column {index} not found in the dataset."
                    )

            if y is not None and df.index.name in (c.name for c in get_cols(y)):
                raise ValueError(
                    "Invalid value for the index parameter. The index column "
                    f"can not be the same as the target column, got {df.index.name}."
                )

            if df.index.duplicated().any():
                raise ValueError(
                    "Invalid value for the index parameter. There are duplicate indices "
                    "in the dataset. Use index=False to reset the index to RangeIndex."
                )

            return df

        def _no_data_sets(
            X: pd.DataFrame,
            y: Pandas,
        ) -> tuple[DataContainer, pd.DataFrame | None]:
            """Generate data sets from one dataset.

            Additionally, assigns an index, shuffles the data, selects
            a subsample if `n_rows` is specified and split into sets in
            a stratified fashion.

            Parameters
            ----------
            X: pd.DataFrame
                Feature set with shape=(n_samples, n_features).

            y: pd.Series or pd.DataFrame
                Target column(s) corresponding to `X`.

            Returns
            -------
            DataContainer
                Train and test sets.

            pd.DataFrame or None
                Holdout data set. Returns None if not specified.

            """
            data = merge(X, y)

            # Shuffle the dataset
            if not 0 < self._config.n_rows <= len(data):
                raise ValueError(
                    "Invalid value for the n_rows parameter. Value should "
                    f"lie between 0 and len(X)={len(data)}, got {self._config.n_rows}."
                )
            data = _subsample(data)

            if isinstance(index, sequence_t):
                if len(index) != len(data):
                    raise IndexError(
                        "Invalid value for the index parameter. Length of index "
                        f"({len(index)}) doesn't match that of the dataset ({len(data)})."
                    )
                data.index = index

            if len(data) < 5:
                raise ValueError(
                    f"The length of the dataset can't be <5, got {len(data)}. "
                    "Make sure n_rows=1 for small datasets."
                )

            if not 0 < self._config.test_size < len(data):
                raise ValueError(
                    "Invalid value for the test_size parameter. Value "
                    f"should lie between 0 and len(X), got {self._config.test_size}."
                )

            # Define test set size
            if self._config.test_size < 1:
                test_size = max(1, int(self._config.test_size * len(data)))
            else:
                test_size = self._config.test_size

            try:
                # Define holdout set size
                if self._config.holdout_size:
                    if self._config.holdout_size < 1:
                        holdout_size = max(1, int(self._config.holdout_size * len(data)))
                    else:
                        holdout_size = self._config.holdout_size

                    if not 0 <= holdout_size <= len(data) - test_size:
                        raise ValueError(
                            "Invalid value for the holdout_size parameter. "
                            "Value should lie between 0 and len(X) - len(test), "
                            f"got {self._config.holdout_size}."
                        )

                    data, holdout = train_test_split(
                        data,
                        test_size=holdout_size,
                        random_state=self.random_state,
                        shuffle=self._config.shuffle,
                        stratify=self._config.get_stratify_columns(data, y),
                    )
                else:
                    holdout = None

                train, test = train_test_split(
                    data,
                    test_size=test_size,
                    random_state=self.random_state,
                    shuffle=self._config.shuffle,
                    stratify=self._config.get_stratify_columns(data, y),
                )

                complete_set = _set_index(pd.concat([train, test, holdout]), y, index)

                container = DataContainer(
                    data=(data := complete_set.iloc[: len(data)]),
                    train_idx=data.index[:-len(test)],
                    test_idx=data.index[-len(test):],
                    n_targets=n_cols(y),
                )

            except ValueError as ex:
                # Clarify common error with stratification for multioutput tasks
                if "least populated class" in str(ex) and isinstance(y, pd.DataFrame):
                    raise ValueError(
                        "Stratification for multioutput tasks is applied over all target "
                        "columns, which results in a least populated class that has only "
                        "one member. Either select only one column to stratify over, or "
                        "set the parameter stratify=False."
                    ) from ex
                else:
                    raise ex

            if holdout is not None:
                holdout = complete_set.iloc[len(data):]

            return container, holdout

        def _has_data_sets(
            X_train: pd.DataFrame,
            y_train: Pandas,
            X_test: pd.DataFrame,
            y_test: Pandas,
            X_holdout: pd.DataFrame | None = None,
            y_holdout: Pandas | None = None,
        ) -> tuple[DataContainer, pd.DataFrame | None]:
            """Generate data sets from provided sets.

            Additionally, assigns an index, shuffles the data and
            selects a subsample if `n_rows` is specified.

            Parameters
            ----------
            X_train: pd.DataFrame
                Training set.

            y_train: pd.Series or pd.DataFrame
                Target column(s) corresponding to `X`_train.

            X_test: pd.DataFrame
                Test set.

            y_test: pd.Series or pd.DataFrame
                Target column(s) corresponding to `X`_test.

            X_holdout: pd.DataFrame or None, default=None
                Holdout set. Can be None if not provided by the user.

            y_holdout: pd.Series, pd.DataFrame or None, default=None
                Target column(s) corresponding to `X`_holdout.

            Returns
            -------
            DataContainer
                Train and test sets.

            pd.DataFrame or None
                Holdout data set. Returns None if not specified.

            """
            train = merge(X_train, y_train)
            test = merge(X_test, y_test)
            if X_holdout is None:
                holdout = None
            else:
                holdout = merge(X_holdout, y_holdout)

            if not train.columns.equals(test.columns):
                raise ValueError("The train and test set do not have the same columns.")

            if holdout is not None:
                if not train.columns.equals(holdout.columns):
                    raise ValueError(
                        "The holdout set does not have the "
                        "same columns as the train and test set."
                    )

            if self._config.n_rows <= 1:
                train = _subsample(train)
                test = _subsample(test)
                if holdout is not None:
                    holdout = _subsample(holdout)
            else:
                raise ValueError(
                    "Invalid value for the n_rows parameter. Value must "
                    "be <1 when the train and test sets are provided."
                )

            # If the index is a sequence, assign it before shuffling
            if isinstance(index, sequence_t):
                len_data = len(train) + len(test)
                if holdout is not None:
                    len_data += len(holdout)

                if len(index) != len_data:
                    raise IndexError(
                        "Invalid value for the index parameter. Length of index "
                        f"({len(index)}) doesn't match that of the data sets ({len_data})."
                    )
                train.index = index[: len(train)]
                test.index = index[len(train): len(train) + len(test)]
                if holdout is not None:
                    holdout.index = index[-len(holdout):]

            complete_set = _set_index(pd.concat([train, test, holdout]), y_test, index)

            container = DataContainer(
                data=(data := complete_set.iloc[:len(train) + len(test)]),
                train_idx=data.index[: len(train)],
                test_idx=data.index[-len(test):],
                n_targets=n_cols(y_train),
            )

            if holdout is not None:
                holdout = complete_set.iloc[len(train) + len(test):]

            return container, holdout

        # Process input arrays ===================================== >>

        if len(arrays) == 0:
            if self.branch._container:
                return self.branch._data, self.branch._holdout
            elif self._goal is Goal.forecast and not isinstance(y, Int | str):
                # arrays=() and y=y for forecasting
                sets = _no_data_sets(*self._check_input(y=y))
            else:
                raise ValueError(
                    "The data arrays are empty! Provide the data to run the pipeline "
                    "successfully. See the documentation for the allowed formats."
                )

        elif len(arrays) == 1:
            # X or y for forecasting
            sets = _no_data_sets(*self._check_input(arrays[0], y=y))

        elif len(arrays) == 2:
            if isinstance(arrays[0], tuple) and len(arrays[0]) == len(arrays[1]) == 2:
                # (X_train, y_train), (X_test, y_test)
                X_train, y_train = self._check_input(arrays[0][0], arrays[0][1])
                X_test, y_test = self._check_input(arrays[1][0], arrays[1][1])
                sets = _has_data_sets(X_train, y_train, X_test, y_test)
            elif isinstance(arrays[1], (*int_t, str)) or n_cols(arrays[1]) == 1:
                if self._goal is not Goal.forecast:
                    # X, y
                    sets = _no_data_sets(*self._check_input(arrays[0], arrays[1]))
                else:
                    # train, test for forecast
                    X_train, y_train = self._check_input(y=arrays[0])
                    X_test, y_test = self._check_input(y=arrays[1])
                    sets = _has_data_sets(X_train, y_train, X_test, y_test)
            else:
                # train, test
                X_train, y_train = self._check_input(arrays[0], y=y)
                X_test, y_test = self._check_input(arrays[1], y=y)
                sets = _has_data_sets(X_train, y_train, X_test, y_test)

        elif len(arrays) == 3:
            if len(arrays[0]) == len(arrays[1]) == len(arrays[2]) == 2:
                # (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)
                X_train, y_train = self._check_input(arrays[0][0], arrays[0][1])
                X_test, y_test = self._check_input(arrays[1][0], arrays[1][1])
                X_hold, y_hold = self._check_input(arrays[2][0], arrays[2][1])
                sets = _has_data_sets(X_train, y_train, X_test, y_test, X_hold, y_hold)
            else:
                # train, test, holdout
                X_train, y_train = self._check_input(arrays[0], y=y)
                X_test, y_test = self._check_input(arrays[1], y=y)
                X_hold, y_hold = self._check_input(arrays[2], y=y)
                sets = _has_data_sets(X_train, y_train, X_test, y_test, X_hold, y_hold)

        elif len(arrays) == 4:
            # X_train, X_test, y_train, y_test
            X_train, y_train = self._check_input(arrays[0], arrays[2])
            X_test, y_test = self._check_input(arrays[1], arrays[3])
            sets = _has_data_sets(X_train, y_train, X_test, y_test)

        elif len(arrays) == 6:
            # X_train, X_test, X_holdout, y_train, y_test, y_holdout
            X_train, y_train = self._check_input(arrays[0], arrays[3])
            X_test, y_test = self._check_input(arrays[1], arrays[4])
            X_hold, y_hold = self._check_input(arrays[2], arrays[5])
            sets = _has_data_sets(X_train, y_train, X_test, y_test, X_hold, y_hold)

        else:
            raise ValueError("Invalid data arrays. See the documentation for the allowed formats.")

        if self._goal.name == "forecast":
            # For forecasting, check if index complies with sktime's standard
            valid, msg, _ = check_is_mtype(
                obj=pd.DataFrame(pd.concat([sets[0].data, sets[1]])),
                mtype="pd.DataFrame",
                return_metadata=True,
                var_name="the dataset",
            )

            if not valid:
                raise ValueError(msg)
        else:
            # Else check for duplicate indices
            if pd.concat([sets[0].data, sets[1]]).index.duplicated().any():
                raise ValueError(
                    "Duplicate indices found in the dataset. "
                    "Try initializing atom using `index=False`."
                )

        return sets

    def _get_models(
        self,
        models: ModelsSelector = None,
        *,
        ensembles: Bool = True,
        branch: Branch | None = None,
    ) -> list[Model]:
        """Get models.

        Models can be selected by name, index or regex pattern. If a
        string is provided, use `+` to select multiple models and `!`
        to exclude them. Models cannot be included and excluded in
        the same call. The input is case-insensitive.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to select. If None, it returns all models.

        ensembles: bool, default=True
            Whether to include ensemble models in the output. If False,
            they are silently excluded from any return.

        branch: Branch or None, default=None
            Force returned models to have been fitted on this branch,
            else raises an exception. If None, this filter is ignored.

        Returns
        -------
        list
            Selected models.

        """
        inc: list[Model] = []
        exc: list[Model] = []
        if models is None:
            inc = self._models.values()
        elif isinstance(models, segment_t):
            inc = get_segment(self._models, models)
        else:
            for model in lst(models):
                if isinstance(model, int_t):
                    try:
                        inc.append(self._models[model])
                    except KeyError:
                        raise IndexError(
                            f"Invalid value for the models parameter. Value {model} is "
                            f"out of range. There are {len(self._models)} models."
                        ) from None
                elif isinstance(model, str):
                    for mdl in model.split("+"):
                        array = inc
                        if mdl.startswith("!") and mdl not in self._models:
                            array = exc
                            mdl = mdl[1:]

                        if mdl.lower() == "winner" and self.winner:
                            array.append(self.winner)
                        elif matches := [
                            m for m in self._models if re.fullmatch(mdl, m.name, re.I)
                        ]:
                            array.extend(matches)
                        else:
                            raise ValueError(
                                "Invalid value for the models parameter. Could "
                                f"not find any model that matches {mdl}. The "
                                f"available models are: {', '.join(self._models.keys())}."
                            )
                elif isinstance(model, Model):
                    inc.append(model)

        if len(inc) + len(exc) == 0:
            raise ValueError(
                "Invalid value for the models parameter, "
                f"got {models}. No models were selected."
            )
        elif inc and exc:
            raise ValueError(
                "Invalid value for the models parameter. You can either "
                "include or exclude models, not combinations of these."
            )
        elif exc:
            # If models were excluded with `!`, select all but those
            inc = [m for m in self._models if m not in exc]

        if not ensembles:
            inc = [m for m in inc if m.acronym not in ("Stack", "Vote")]

        if branch and not all(m.branch is branch for m in inc):
            raise ValueError(
                "Invalid value for the models parameter. Not "
                f"all models have been fitted on {branch}."
            )

        return list(dict.fromkeys(inc))  # Avoid duplicates

    def _delete_models(self, models: str | Model | Sequence[str | Model]):
        """Delete models.

        Remove models from the instance. All attributes are deleted
        except for `errors`. If all models are removed, the metric is
        reset.

        Parameters
        ----------
        models: str, Model or sequence
            Model(s) to delete.

        """
        for model in lst(models):
            if model in self._models:
                self._models.remove(model)

        # If no models, reset the metric
        if not self._models:
            self._metric = ClassMap()

    @crash
    def available_models(self, **kwargs) -> pd.DataFrame:
        """Give an overview of the available predefined models.

        Parameters
        ----------
        **kwargs
            Filter the returned models providing any of the column as
            keyword arguments, where the value is the desired filter,
            e.g., `accepts_sparse=True`, to get all models that accept
            sparse input or `supports_engines="cuml"` to get all models
            that support the [cuML][] engine.

        Returns
        -------
        pd.DataFrame
            Tags of the available [predefined models][]. The columns
            depend on the task, but can include:

            - **acronym:** Model's acronym (used to call the model).
            - **fullname:** Name of the model's class.
            - **estimator:** Name of the model's underlying estimator.
            - **module:** The estimator's module.
            - **handles_missing:** Whether the model can handle missing
              values without preprocessing. If False, consider using the
              [Imputer][] class before training the models.
            - **needs_scaling:** Whether the model requires feature scaling.
              If True, [automated feature scaling][] is applied.
            - **accepts_sparse:** Whether the model accepts [sparse input][sparse-datasets].
            - **uses_exogenous:** Whether the model uses [exogenous variables][].
            - **multiple_seasonality:** Whether the model can handle more than
              one [seasonality period][seasonality].
            - **native_multilabel:** Whether the model has native support
              for [multilabel][] tasks.
            - **native_multioutput:** Whether the model has native support
              for [multioutput tasks][].
            - **validation:** Whether the model has [in-training validation][].
            - **supports_engines:** Engines supported by the model.

        """
        rows = []
        for model in MODELS:
            m = model(goal=self._goal, branches=self._branches)
            if self._goal.name in m._estimators:
                tags = m.get_tags()

                for key, value in kwargs.items():
                    k = tags.get(key)
                    if isinstance(value, bool_t) and value is not bool(k):
                        break
                    elif isinstance(value, str) and not re.search(value, k, re.I):
                        break
                else:
                    rows.append(tags)

        return pd.DataFrame(rows)

    @composed(crash, method_to_log)
    def clear(self):
        """Reset attributes and clear cache from all models.

        Reset certain model attributes to their initial state, deleting
        potentially large data arrays. Use this method to free some
        memory before [saving][self-save] the instance. The affected
        attributes are:

        - [In-training validation][] scores
        - [Shap values][shap]
        - [App instance][adaboost-create_app]
        - [Dashboard instance][adaboost-create_dashboard]
        - Calculated [holdout data sets][data-sets]

        """
        for model in self._models:
            model.clear()

    @composed(crash, method_to_log, beartype)
    def delete(self, models: ModelsSelector = None):
        """Delete models.

        If all models are removed, the metric is reset. Use this method
        to drop unwanted models from the pipeline or to free some memory
        before [saving][self-save]. Deleted models are not removed from
        any active [mlflow experiment][tracking].

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to delete. If None, all models are deleted.

        """
        self._log(f"Deleting {len(models := self._get_models(models))} models...", 1)
        for m in models:
            self._delete_models(m.name)
            self._log(f" --> Model {m.name} successfully deleted.", 1)

    @composed(crash, beartype)
    def evaluate(
        self,
        metric: MetricConstructor = None,
        rows: RowSelector = "test",
        *,
        threshold: FloatZeroToOneExc | Sequence[FloatZeroToOneExc] = 0.5,
        as_frame: Bool = False,
    ) -> Any | pd.DataFrame:
        """Get all models' scores for the provided metrics.

        Parameters
        ----------
        metric: str, func, scorer, sequence or None, default=None
            Metric to calculate. If None, it returns an overview of
            the most common metrics per task.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Selection of rows][row-and-column-selection] to calculate
            metric on.

        threshold: float or sequence, default=0.5
            Threshold between 0 and 1 to convert predicted probabilities
            to class labels. Only used when:

            - The task is binary or [multilabel][] classification.
            - The model has a `predict_proba` method.
            - The metric evaluates predicted probabilities.

            For multilabel classification tasks, it's possible to
            provide a sequence of thresholds (one per target column).
            The same threshold per target column is applied to all
            models.

        as_frame: bool, default=False
            Whether to return the scores as a pd.DataFrame. If False, a
            `pandas.io.formats.style.Styler` object is returned, which
            has a `_repr_html_` method defined, so it is rendered
            automatically in a notebook. The highest score per metric
            is highlighted.

        Returns
        -------
        [Styler][] or pd.DataFrame
            Scores of the models.

        """
        check_is_fitted(self)

        df = pd.DataFrame([m.evaluate(metric, rows, threshold=threshold) for m in self._models])

        if len(self._models) == 1 or as_frame:
            return df
        else:
            return df.style.highlight_max(props="background-color: lightgreen")

    @composed(crash, beartype)
    def export_pipeline(self, model: str | Model | None = None) -> Pipeline:
        """Export the internal pipeline.

        This method returns a deepcopy of the branch's pipeline.
        Optionally, you can add a model as final estimator. The
        returned pipeline is already fitted on the training set.

        Parameters
        ----------
        model: str, Model or None, default=None
            Model for which to export the pipeline. If the model used
            [automated feature scaling][], the [Scaler][] is added to
            the pipeline. If None, the pipeline in the current branch
            is exported (without any model).

        Returns
        -------
        [Pipeline][]
            Current branch as a sklearn-like Pipeline object.

        """
        if model:
            return self._get_models(model)[0].export_pipeline()
        else:
            return deepcopy(self.pipeline)

    @available_if(has_task("classification"))
    @composed(crash, beartype)
    def get_class_weight(
        self,
        rows: RowSelector = "train",
    ) -> dict[Hashable, float] | dict[str, dict[Hashable, float]]:
        """Return class weights for a balanced data set.

        Statistically, the class weights re-balance the data set so
        that the sampled data set represents the target population
        as closely as possible. The returned weights are inversely
        proportional to the class frequencies in the selected rows.

        Parameters
        ----------
        rows: hashable, segment, sequence or dataframe, default="train"
            [Selection of rows][row-and-column-selection] for which to
            get the weights.

        Returns
        -------
        dict
            Classes with the corresponding weights. A dict of dicts is
            returned for [multioutput tasks][].

        """

        def get_weights(col: pd.Series) -> dict[Hashable, float]:
            """Get the class weights for one column.

            Parameters
            ----------
            col: pd.Series
                Column to get the weights from.

            Returns
            -------
            dict
                Class weights.

            """
            counts = col.value_counts().sort_index()
            return {n: divide(counts.iloc[0], v, 3) for n, v in counts.items()}

        _, y = self.branch._get_rows(rows, return_X_y=True)

        if self.task.is_multioutput:
            return {str(col.name): get_weights(col) for col in get_cols(y)}
        else:
            return get_weights(y)

    @available_if(has_task("classification"))
    @composed(crash, beartype)
    def get_sample_weight(self, rows: RowSelector = "train") -> pd.Series:
        """Return sample weights for a balanced data set.

        The returned weights are inversely proportional to the class
        frequencies in the selected data set. For [multioutput tasks][],
        the weights of each column of `y` will be multiplied.

        Parameters
        ----------
        rows: hashable, segment, sequence or dataframe, default="train"
            [Selection of rows][row-and-column-selection] for which to
            get the weights.

        Returns
        -------
        pd.Series
            Sequence of weights with shape=(n_samples,).

        """
        _, y = self.branch._get_rows(rows, return_X_y=True)
        weights = compute_sample_weight("balanced", y=y)
        return pd.Series(weights, name="sample_weight").round(3)

    @available_if(has_task("forecast"))
    @composed(crash, beartype)
    def get_seasonal_period(
        self,
        max_sp: IntLargerOne | None = None,
        harmonics: HarmonicsSelector | None = None,
        target: TargetSelector = 0,
    ) -> int | list[int]:
        """Get the seasonal periods of the time series.

        Use the data in the training set to calculate the seasonal
        period. The data is internally differentiated before the
        seasonality is detected using ACF.

        !!! tip
            Read more about seasonality in the [user guide][seasonality].

        Parameters
        ----------
        max_sp: int or None, default=None
            Maximum seasonal period to consider. If None, the maximum
            period is given by `(len(y_train) - 1) // 2`.

        harmonics: str or None, default=None
            Defines the strategy on how to deal with harmonics from the
            detected seasonal periods. Choose from the following options:

            - None: The detected seasonal periods are left unchanged
              (no harmonic removal).
            - "drop": Remove all harmonics.
            - "raw_strength": Keep the highest order harmonics, maintaining
              the order of significance.
            - "harmonic_strength": Replace seasonal periods with their highest
              harmonic.

            E.g., if the detected seasonal periods in strength order are
            `[2, 3, 4, 7, 8]` (note that 4 and 8 are harmonics of 2), then:

            - If "drop", result=[2, 3, 7]
            - If "raw_strength", result=[3, 7, 8]
            - If "harmonic_strength", result=[8, 3, 7]

        target: int or str, default=0
            Target column to look at. Only for [multivariate][] tasks.

        Returns
        -------
        int or list of int
            Seasonal periods, ordered by significance.

        """
        yt = self.dataset[self.branch._get_target(target, only_columns=True)]
        max_sp = max_sp or (len(yt) - 1) // 2

        for _ in np.arange(ndiffs(yt)):
            yt = Differencer().fit_transform(yt)

        acf = SeasonalityACF(nlags=max_sp).fit(pd.DataFrame(yt))
        seasonal_periods = acf.get_fitted_params().get("sp_significant")

        if harmonics and len(seasonal_periods) > 1:
            # Create a dictionary of the seasonal periods and their harmonics
            harmonic_dict: dict[int, list[int]] = {}
            for sp in seasonal_periods:
                for k in harmonic_dict:
                    if sp % k == 0:
                        harmonic_dict[k].append(sp)
                        break
                else:
                    harmonic_dict[sp] = []

            # For periods without harmonics, simplify operations
            # by setting the value of the key to itself
            harmonic_dict = {k: (v or [k]) for k, v in harmonic_dict.items()}

            if harmonics == "drop":
                seasonal_periods = list(harmonic_dict.keys())
            elif harmonics == "raw_strength":
                seasonal_periods = [
                    sp for sp in seasonal_periods
                    if any(max(v) == sp for v in harmonic_dict.values())
                ]
            elif harmonics == "harmonic_strength":
                seasonal_periods = [max(v) for v in harmonic_dict.values()]

        if not (seasonal_periods := [int(sp) for sp in seasonal_periods if sp <= max_sp]):
            raise ValueError(
                "No seasonal periods were detected. Try decreasing the max_sp parameter."
            )

        return flt(seasonal_periods)

    @composed(crash, method_to_log, beartype)
    def merge(self, other: BaseRunner, /, suffix: str = "2"):
        """Merge another instance of the same class into this one.

        Branches, models, metrics and attributes of the other instance
        are merged into this one. If there are branches and/or models
        with the same name, they are merged adding the `suffix`
        parameter to their name. The errors and missing attributes are
        extended with those of the other instance. It's only possible
        to merge two instances if they are initialized with the same
        dataset and trained with the same metric.

        Parameters
        ----------
        other: Runner
            Instance with which to merge. Should be of the same class
            as self.

        suffix: str, default="2"
            Branches and models with conflicting names are merged adding
            `suffix` to the end of their names.

        """
        if other.__class__.__name__ != self.__class__.__name__:
            raise TypeError(
                "Invalid class for the other parameter. Expecting a "
                f"{self.__class__.__name__} instance, got {other.__class__.__name__}."
            )

        # Check that both instances have the same original dataset
        if not self.og._data.data.equals(other.og._data.data):
            raise ValueError(
                "Invalid value for the other parameter. The provided instance "
                "was initialized using a different dataset than this one."
            )

        # Check that both instances have the same metric
        if not self._metric:
            self._metric = other._metric
        elif other.metric and self.metric != other.metric:
            raise ValueError(
                "Invalid value for the other parameter. The provided instance uses "
                f"a different metric ({other.metric}) than this one ({self.metric})."
            )

        self._log("Merging instances...", 1)
        for branch in other._branches:
            self._log(f" --> Merging branch {branch.name}.", 1)
            if branch.name in self._branches:
                branch._name = f"{branch.name}{suffix}"
            self._branches.branches[branch.name] = branch

        for model in other._models:
            self._log(f" --> Merging model {model.name}.", 1)
            if model.name in self._models:
                model._name = f"{model.name}{suffix}"
            self._models[model.name] = model

        self._log(" --> Merging attributes.", 1)
        if hasattr(self, "missing"):
            self.missing.extend([x for x in other.missing if x not in self.missing])

    @composed(crash, method_to_log, beartype)
    def save(self, filename: str | Path = "auto", *, save_data: Bool = True):
        """Save the instance to a pickle file.

        Parameters
        ----------
        filename: str or Path, default="auto"
            Filename or [pathlib.Path][] of the file to save. Use
            "auto" for automatic naming.

        save_data: bool, default=True
            Whether to save the dataset with the instance. This
            parameter is ignored if the method is not called from atom.
            If False, add the data to the [load][atomclassifier-load]
            method to reload the instance.

        """
        if not save_data:
            data = {}
            if (og := self._branches.og).name not in self._branches:
                self._branches._og._container = None
            for branch in self._branches:
                data[branch.name] = {
                    "_data": deepcopy(branch._container),
                    "_holdout": deepcopy(branch._holdout),
                    "holdout": branch.__dict__.pop("holdout", None),  # Clear cached holdout
                }
                branch._container = None
                branch._holdout = None

        if (path := Path(filename)).suffix != ".pkl":
            path = path.with_suffix(".pkl")

        if path.name == "auto.pkl":
            path = path.with_name(f"{self.__class__.__name__}.pkl")

        with open(path, "wb") as f:
            pickle.settings["recurse"] = True
            pickle.dump(self, f)

        # Restore the data to the attributes
        if not save_data:
            if og.name not in self._branches:
                self._branches._og._container = og._container
            for branch in self._branches:
                branch._container = data[branch.name]["_data"]
                branch._holdout = data[branch.name]["_holdout"]
                if data[branch.name]["holdout"] is not None:
                    branch.__dict__["holdout"] = data[branch.name]["holdout"]

        self._log(f"{self.__class__.__name__} successfully saved.", 1)

    @composed(crash, method_to_log, beartype)
    def stacking(
        self,
        models: Segment | Sequence[ModelSelector] | None = None,
        name: str = "Stack",
        *,
        train_on_test: Bool = False,
        **kwargs,
    ):
        """Add a [Stacking][] model to the pipeline.

        !!! warning
            Combining models trained on different branches into one
            ensemble is not allowed and will raise an exception.

        Parameters
        ----------
        models: segment, sequence or None, default=None
            Models that feed the stacking estimator. The models must
            have been fitted on the current branch.

        name: str, default="Stack"
            Name of the model. The name is always presided with the
            model's acronym: `Stack`.

        train_on_test: bool, default=False
            Whether to train the final estimator of the stacking model
            on the test set instead of the training set. Note that
            training it on the training set (default option) means there
            is a high risk of overfitting. It's recommended to use this
            option if you have another, independent set for testing
            ([holdout set][data-sets]).

        **kwargs
            Additional keyword arguments for one of these estimators.

            - For classification tasks: [StackingClassifier][].
            - For regression tasks: [StackingRegressor][].
            - For forecast tasks: [StackingForecaster][].

            !!! tip
                The model's acronyms can be used for the `final_estimator`
                parameter, e.g., `atom.stacking(final_estimator="LR")`.

        """
        check_is_fitted(self)
        models_c = self._get_models(models, ensembles=False, branch=self.branch)

        if len(models_c) < 2:
            raise ValueError(
                "Invalid value for the models parameter. A Stacking model should "
                f"contain at least two underlying estimators, got only {models_c[0]}."
            )

        if not name.lower().startswith("stack"):
            name = f"Stack{name}"

        if name in self._models:
            raise ValueError(
                "Invalid value for the name parameter. It seems a model with "
                f"the name {name} already exists. Add a different name to "
                "train multiple Stacking models within the same instance."
            )

        kw_model = {
            "goal": self._goal,
            "config": self._config,
            "branches": self._branches,
            "metric": self._metric,
            **{attr: getattr(self, attr) for attr in BaseTransformer.attrs},
        }

        # The parameter name is different in sklearn and sktime
        regressor = "regressor" if self.task.is_forecast else "final_estimator"
        if isinstance(kwargs.get(regressor), str):
            if kwargs["final_estimator"] not in MODELS:
                valid_models = [m.acronym for m in MODELS if self._goal.name in m._estimators]
                raise ValueError(
                    f"Invalid value for the {regressor} parameter. Unknown model: "
                    f"{kwargs[regressor]}. Choose from: {', '.join(valid_models)}."
                )
            else:
                model = MODELS[kwargs[regressor]](**kw_model)
                if self._goal.name not in model._estimators:
                    raise ValueError(
                        f"Invalid value for the {regressor} parameter. Model "
                        f"{model.fullname} can not perform {self.task} tasks."
                    )

                kwargs[regressor] = model._get_est({})

        self._models.append(Stacking(models=models_c, name=name, **kw_model))
        self[name]._est_params = kwargs if self.task.is_forecast else {"cv": "prefit"} | kwargs

        if train_on_test:
            self[name].fit(self.X_test, self.y_test)
        else:
            self[name].fit()

    @composed(crash, method_to_log, beartype)
    def voting(
        self,
        models: Segment | Sequence[ModelSelector] | None = None,
        name: str = "Vote",
        **kwargs,
    ):
        """Add a [Voting][] model to the pipeline.

        !!! warning
            Combining models trained on different branches into one
            ensemble is not allowed and will raise an exception.

        Parameters
        ----------
        models: segment, sequence or None, default=None
            Models that feed the stacking estimator. The models must have
            been fitted on the current branch.

        name: str, default="Vote"
            Name of the model. The name is always presided with the
            model's acronym: `Vote`.

        **kwargs
            Additional keyword arguments for one of these estimators.

            - For classification tasks: [VotingClassifier][].
            - For regression tasks: [VotingRegressor][].
            - For forecast tasks: [EnsembleForecaster][].

        """
        check_is_fitted(self)
        models_c = self._get_models(models, ensembles=False, branch=self.branch)

        if len(models_c) < 2:
            raise ValueError(
                "Invalid value for the models parameter. A Voting model should "
                f"contain at least two underlying estimators, got only {models_c[0]}."
            )

        if not name.lower().startswith("vote"):
            name = f"Vote{name}"

        if name in self._models:
            raise ValueError(
                "Invalid value for the name parameter. It seems a model with "
                f"the name {name} already exists. Add a different name to "
                "train multiple Voting models within the same instance."
            )

        if kwargs.get("voting") == "soft":
            for m in models_c:
                if not hasattr(m.estimator, "predict_proba"):
                    raise ValueError(
                        "Invalid value for the voting parameter. If "
                        "'soft', all models in the ensemble should have "
                        f"a predict_proba method, got {m.name}."
                    )

        self._models.append(
            Voting(
                models=models_c,
                name=name,
                goal=self._goal,
                config=self._config,
                branches=self._branches,
                metric=self._metric,
                **{attr: getattr(self, attr) for attr in BaseTransformer.attrs},
            )
        )
        self[name]._est_params = kwargs
        self[name].fit()

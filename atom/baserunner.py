# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseRunner class.

"""

import re
import tempfile
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import mlflow
import pandas as pd
from joblib.memory import Memory
from sklearn.base import clone
from sklearn.multioutput import (
    ClassifierChain, MultiOutputClassifier, MultiOutputRegressor,
    RegressorChain,
)
from typeguard import typechecked

from atom.basemodel import BaseModel
from atom.branch import Branch
from atom.models import MODELS, Stacking, Voting
from atom.pipeline import Pipeline
from atom.utils import (
    DF_ATTRS, FLOAT, INT, SEQUENCE_TYPES, CustomDict, Model, Predictor,
    check_is_fitted, composed, crash, divide, flt, get_best_score, get_pl_name,
    get_versions, lst, method_to_log,
)


class BaseRunner:
    """Base class for runners.

    Contains shared attributes and methods for the atom and trainer
    classes. Implements magic methods, mlflow tracking properties,
    utility properties, prediction methods and utility methods.

    """

    # Tracking parameters for mlflow
    _tracking_params = dict(
        log_ht=True,
        log_model=True,
        log_plots=True,
        log_data=False,
        log_pipeline=False,
    )

    def __init__(self):
        self._multioutput = "auto"

    def __getstate__(self) -> dict:
        # Store an extra attribute with the package versions
        return {**self.__dict__, "_versions": get_versions(self._models)}

    def __setstate__(self, state: dict):
        versions = state.pop("_versions", None)
        self.__dict__.update(state)

        # Check that all package versions match or raise a warning
        if versions:
            current_versions = get_versions(state["_models"])
            for key, value in current_versions.items():
                if versions[key] != value:
                    self.log(
                        f"The loaded instance used the {key} package with version "
                        f"{versions[key]} while the version in this environment is "
                        f"{value}.", 1, severity="warning"
                    )

    def __getattr__(self, item: str) -> Any:
        if item in self.__dict__.get("_branches").min("og"):
            return self._branches[item]  # Get branch
        elif item in self.branch._get_attrs():
            return getattr(self.branch, item)  # Get attr from branch
        elif self.__dict__.get("_models").get(item.lower()):
            return self._models[item.lower()]  # Get model subclass
        elif item in self.branch.columns:
            return self.branch.dataset[item]  # Get column from dataset
        elif item in DF_ATTRS:
            return getattr(self.branch.dataset, item)  # Get attr from dataset
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'."
            )

    def __setattr__(self, item: str, value: Any):
        if item != "holdout" and isinstance(getattr(Branch, item, None), property):
            setattr(self.branch, item, value)
        else:
            super().__setattr__(item, value)

    def __delattr__(self, item: str):
        if item in self._branches:
            self.branch.__delete__(self._branches[item])
        elif item in self._models:
            self.delete(item)
        else:
            super().__delattr__(item)

    def __len__(self) -> int:
        return len(self.dataset)

    def __contains__(self, item: str) -> bool:
        if self.dataset is None:
            return False
        else:
            return item in self.dataset

    def __getitem__(self, item: Union[INT, str, list]) -> Any:
        if self.dataset is None:
            raise RuntimeError(
                "This instance has no dataset annexed to it. "
                "Use the run method before calling __getitem__."
            )
        elif isinstance(item, int):
            return self.dataset[self.columns[item]]
        elif isinstance(item, str):
            if item in self._branches.min("og"):
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
        elif isinstance(item, list):
            return self.dataset[item]  # Get subset of dataset
        else:
            raise TypeError(
                f"{self.__class__.__name__} is only "
                "subscriptable with types int, str or list."
            )

    # Tracking properties ========================================== >>

    @property
    def log_ht(self) -> bool:
        """Whether to track every trial of the hyperparameter tuning."""
        return self._tracking_params["log_ht"]

    @log_ht.setter
    @typechecked
    def log_ht(self, value: bool):
        self._tracking_params["log_ht"] = value

    @property
    def log_model(self) -> bool:
        """Whether to save the model's estimator after fitting."""
        return self._tracking_params["log_model"]

    @log_model.setter
    @typechecked
    def log_model(self, value: bool):
        self._tracking_params["log_model"] = value

    @property
    def log_plots(self) -> bool:
        """Whether to save plots as artifacts."""
        return self._tracking_params["log_plots"]

    @log_plots.setter
    @typechecked
    def log_plots(self, value: bool):
        self._tracking_params["log_plots"] = value

    @property
    def log_data(self) -> bool:
        """Whether to save the train and test sets."""
        return self._tracking_params["log_data"]

    @log_data.setter
    @typechecked
    def log_data(self, value: bool):
        self._tracking_params["log_data"] = value

    @property
    def log_pipeline(self) -> bool:
        """Whether to save the model's pipeline."""
        return self._tracking_params["log_pipeline"]

    @log_pipeline.setter
    @typechecked
    def log_pipeline(self, value: bool):
        self._tracking_params["log_pipeline"] = value

    # Utility properties =========================================== >>

    @property
    def _is_multioutput(self) -> bool:
        """Return whether the task is multilabel or multioutput."""
        return any(task in self.task for task in ("multilabel", "multioutput"))

    @property
    def branch(self) -> Branch:
        """Current active branch.

        Use the property's `@setter` to change from current branch or
        to create a new one. If the value is the name of an existing
        branch, switch to that one. Else, create a new branch using
        that name. The new branch is split from the current branch. Use
        `__from__` to split the new branch from any other existing
        branch. Read more in the [user guide][branches].

        """
        return self._branches[self._current]

    @branch.deleter
    def branch(self):
        """Delete the current active branch."""
        self.branch.__delete__(self.branch)

    @property
    def multioutput(self) -> Optional[Predictor]:
        """Meta-estimator for [multioutput tasks][].

        This estimator is only used when the model has no native
        support for multioutput tasks. Use the `@setter` to set any
        other meta-estimator (the underlying estimator should be the
        first parameter in the constructor) or set equal to None to
        avoid this wrapper.

        """
        if self._multioutput == "auto":
            if self.task.startswith("multilabel"):
                if self.goal.startswith("class"):
                    return ClassifierChain
                else:
                    return RegressorChain
            else:
                if self.goal.startswith("class"):
                    return MultiOutputClassifier
                else:
                    return MultiOutputRegressor
        else:
            return self._multioutput

    @multioutput.setter
    @typechecked
    def multioutput(self, value: Optional[Union[str, Predictor]]):
        """Assign a new multioutput meta-estimator."""
        if value is None:
            self._multioutput = value
        elif isinstance(value, str):
            if value.lower() != "auto":
                raise ValueError(
                    "Invalid value for the multioutput attribute. Use 'auto' "
                    "for the default meta-estimator, None to ignore it, or "
                    "provide a custom meta-estimator class or instance."
                )
            self._multioutput = "auto"
        elif callable(value):
            self._multioutput = value
        else:
            self._multioutput = clone(value)

    @property
    def models(self) -> Union[str, List[str]]:
        """Name of the model(s)."""
        if isinstance(self._models, CustomDict):
            return flt([model.name for model in self._models.values()])
        else:
            return self._models

    @property
    def metric(self) -> Union[str, List[str]]:
        """Name of the metric(s)."""
        if isinstance(self._metric, CustomDict):
            return flt([metric.name for metric in self._metric.values()])
        else:
            return self._metric

    @property
    def errors(self) -> CustomDict:
        """Errors encountered during model training.

        The key is the model's name and the value is the exception
        object that was raised. Use the `__traceback__` attribute to
        investigate the error.

        """
        return self._errors

    @property
    def winners(self) -> List[Model]:
        """Models ordered by performance.

        Performance is measured as the highest score on the model's
        [`score_bootstrap`][adaboost-score_bootstrap] or
        [`score_test`][adaboost-score_test] attributes, checked in
        that order. For [multi-metric runs][], only the main metric
        is compared.

        """
        if self._models:  # Returns None if not fitted
            return sorted(
                self._models.values(), key=lambda x: get_best_score(x), reverse=True
            )

    @property
    def winner(self) -> Model:
        """Best performing model.

        Performance is measured as the highest score on the model's
        [`score_bootstrap`][adaboost-score_bootstrap] or
        [`score_test`][adaboost-score_test] attributes, checked in
        that order. For [multi-metric runs][], only the main metric
        is compared.

        """
        if self._models:  # Returns None if not fitted
            return self.winners[0]

    @winner.deleter
    def winner(self):
        """[Delete][atomclassifier-delete] the best performing model."""
        if self._models:  # Do nothing if not fitted
            self.delete(self.winner.name)

    @property
    def results(self) -> pd.DataFrame:
        """Overview of the training results.

        All durations are in seconds. Columns include:

        - **score_ht:** Score obtained by the hyperparameter tuning.
        - **time_ht:** Duration of the hyperparameter tuning.
        - **score_train:** Metric score on the train set.
        - **score_test:** Metric score on the test set.
        - **time_fit:** Duration of the model fitting on the train set.
        - **score_bootstrap:** Mean score on the bootstrapped samples.
        - **time_bootstrap:** Duration of the bootstrapping.
        - **time:** Total duration of the model run.

        """

        def frac(m: Model) -> float:
            """Return the fraction of the train set used for the model."""
            n_models = len(m.branch.train) / m._train_idx
            if n_models == int(n_models):
                return round(1.0 / n_models, 2)
            else:
                return round(m._train_idx / len(m.branch.train), 2)

        df = pd.DataFrame(
            data=[m.results for m in self._models.values()],
            columns=self._models[0].results.index if self._models else [],
            index=lst(self.models),
        ).dropna(axis=1, how="all")

        # For sh and ts runs, include the fraction of training set
        if any(m._train_idx != len(m.branch.train) for m in self._models.values()):
            df = df.set_index(
                pd.MultiIndex.from_arrays(
                    arrays=[[frac(m) for m in self._models.values()], self.models],
                    names=["frac", "model"],
                )
            ).sort_index(level=0, ascending=True)

        return df

    # Utility methods ============================================== >>

    def _get_og_branches(self):
        """Return branches containing the original dataset."""
        return [branch for branch in self._branches.values() if branch.pipeline.empty]

    def _get_rows(
        self,
        index: Optional[Union[INT, str, slice, SEQUENCE_TYPES]] = None,
        return_test: bool = True,
        branch: Optional[Branch] = None,
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

        branch: Branch or None, default=None
            Get columns from specified branch. If None, use the current
            branch.

        Returns
        -------
        list
            Indices of the included rows.

        """

        def get_match(idx: str, ex: Optional[ValueError] = None):
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

        if not branch:
            branch = self.branch

        indices = list(branch.dataset.index)
        if branch.holdout is not None:
            indices += list(branch.holdout.index)

        inc, exc = [], []
        if index is None:
            inc = list(branch._idx[2]) if return_test else list(branch.X.index)
        elif isinstance(index, slice):
            inc = indices[index]
        else:
            for idx in lst(index):
                if isinstance(idx, (int, float, str)) and idx in indices:
                    inc.append(idx)
                elif isinstance(idx, int):
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
        columns: Optional[Union[INT, str, slice, SEQUENCE_TYPES]] = None,
        include_target: bool = True,
        return_inc_exc: bool = False,
        only_numerical: bool = False,
        branch: Optional[Branch] = None,
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
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

        branch: Branch or None, default=None
            Get columns from specified branch. If None, use the current
            branch.

        Returns
        -------
        list
            Names of the included columns.

        list
            Names of the excluded columns. Only returned if
            return_inc_exc=True.

        """

        def get_match(col: str, ex: Optional[ValueError] = None):
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

        if not branch:
            branch = self.branch

        # Select dataframe from which to get the columns
        df = branch.dataset if include_target else branch.X

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
                if isinstance(col, int):
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

    def _get_models(
        self,
        models: Optional[Union[INT, str, Model, slice, SEQUENCE_TYPES]] = None,
        ensembles: bool = True,
    ) -> List[str]:
        """Get names of models.

        Models can be selected by name, index or regex pattern. If a
        string is provided, use `+` to select multiple models and `!`
        to exclude them. Models cannot be included and excluded in
        the same call. The input is case-insensitive.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Names or indices of the models to select. If None, it
            returns all models.

        ensembles: bool, default=True
            Whether to include ensemble models in the output. If False,
            they are silently ignored.

        Returns
        -------
        list
            Model names.

        """

        def get_match(model: str, ex: Optional[ValueError] = None):
            """Try to find a match by regex.

            Parameters
            ----------
            model: str
                Regex pattern to match with model names.

            ex: ValueError or None
                Exception to raise if failed (from previous call).

            """
            nonlocal inc, exc

            array = inc
            if model.startswith("!") and model not in options:
                array = exc
                model = model[1:]

            # Find rows using regex matches
            if model.lower() == "winner":
                array.append(self.winner.name)
            elif matches := [m for m in options if re.fullmatch(model, m, re.IGNORECASE)]:
                array.extend(matches)
            else:
                raise ex or ValueError(
                    "Invalid value for the models parameter. Could "
                    f"not find any model that matches {model}. The "
                    f"available models are: {', '.join(options)}."
                )

        options = self._models.min("og")

        inc, exc = [], []
        if models is None:
            inc.extend(options)
        elif isinstance(models, slice):
            inc.extend(options[models])
        else:
            for model in lst(models):
                if isinstance(model, int):
                    try:
                        inc.append(options[model].name)
                    except KeyError:
                        raise ValueError(
                            "Invalid value for the models parameter. Value "
                            f"{model} is out of range for a pipeline with "
                            f"{len(options)} models."
                        )
                elif isinstance(model, str):
                    try:
                        get_match(model)
                    except ValueError as ex:
                        for m in model.split("+"):
                            get_match(m, ex)
                elif isinstance(model, BaseModel):
                    inc.append(model.name)
                else:
                    raise TypeError(
                        f"Invalid type for the models parameter, got {type(model)}. "
                        "Use a model's name, position or instance to select it."
                    )

        if inc and exc:
            raise ValueError(
                "Invalid value for the models parameter. You can either "
                "include or exclude models, not combinations of these."
            )

        if exc:
            # If models were excluded with `!`, select all but those
            inc = [m for m in options if m not in exc]

        if not ensembles:
            inc = [
                model for model in inc
                if not model.startswith("Stack") and not model.startswith("Vote")
            ]

        return list(dict.fromkeys(inc))  # Avoid duplicates

    def _delete_models(self, models: SEQUENCE_TYPES):
        """Delete models.

        Remove models from the instance. All attributes are deleted
        except for `errors`. If all models are removed, the metric is
        reset.

        Parameters
        ----------
        models: sequence
            Names of the models to delete.

        """
        for model in models:
            self._models.pop(model)

        # If no models, reset the metric
        if not self._models:
            self._metric = CustomDict()

    @crash
    def available_models(self) -> pd.DataFrame:
        """Give an overview of the available predefined models.

        Returns
        -------
        pd.DataFrame
            Information about the available [predefined models][]. Columns
            include:

            - **acronym:** Model's acronym (used to call the model).
            - **model:** Name of the model's class.
            - **estimator:** The model's underlying estimator.
            - **module:** The estimator's module.
            - **needs_scaling:** Whether the model requires feature scaling.
            - **accepts_sparse:** Whether the model accepts sparse matrices.
            - **native_multioutput:** Whether the model has native support
              for [multioutput tasks][].
            - **has_validation:** Whether the model has [in-training validation][].
            - **supports_engines:** List of engines supported by the model.

        """
        rows = []
        for model in MODELS.values():
            m = model(self, fast_init=True)
            if self.goal in m._estimators:
                rows.append(
                    {
                        "acronym": m.acronym,
                        "model": m._fullname,
                        "estimator": m._est_class.__name__,
                        "module": m._est_class.__module__.split(".")[0] + m._module,
                        "needs_scaling": m.needs_scaling,
                        "accepts_sparse": m.accepts_sparse,
                        "native_multioutput": m.native_multioutput,
                        "has_validation": bool(m.has_validation),
                        "supports_engines": ", ". join(m.supports_engines),
                    }
                )

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
        - Cached [prediction attributes][]
        - Cached [metric scores][metric]
        - Cached [holdout data sets][data-sets]

        """
        for model in self._models.values():
            model.clear()

    @composed(crash, method_to_log, typechecked)
    def delete(
        self, models: Optional[Union[INT, str, slice, Model, SEQUENCE_TYPES]] = None
    ):
        """Delete models.

        If all models are removed, the metric is reset. Use this method
        to drop unwanted models from the pipeline or to free some memory
        before [saving][self-save]. Deleted models are not removed from
        any active [mlflow experiment][tracking].

        Parameters
        ----------
        models: int, str, slice, Model, sequence or None, default=None
            Models to delete. If None, all models are deleted.

        """
        models = self._get_models(models)
        if not models:
            self.log("No models to delete.", 1)
        else:
            self._delete_models(models)
            if len(models) == 1:
                self.log(f"Model {models[0]} successfully deleted.", 1)
            else:
                self.log(f"Deleting {len(models)} models...", 1)
                for m in models:
                    self.log(f" --> Model {m} successfully deleted.", 1)

    @composed(crash, typechecked)
    def evaluate(
        self,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        threshold: FLOAT = 0.5,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
    ) -> pd.DataFrame:
        """Get all models' scores for the provided metrics.

        Parameters
        ----------
        metric: str, func, scorer, sequence or None, default=None
            Metric to calculate. If None, it returns an overview of
            the most common metrics per task.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

        threshold: float, default=0.5
            Threshold between 0 and 1 to convert predicted probabilities
            to class labels. Only used when:

            - The task is binary classification.
            - The model has a `predict_proba` method.
            - The metric evaluates predicted target values.

        sample_weight: sequence or None, default=None
            Sample weights corresponding to y in `dataset`.

        Returns
        -------
        pd.DataFrame
            Scores of the models.

        """
        check_is_fitted(self, attributes="_models")

        return pd.DataFrame(
            [
                m.evaluate(metric, dataset, threshold, sample_weight)
                for m in self._models.values()
            ]
        )

    @composed(crash, typechecked)
    def export_pipeline(
        self,
        model: Optional[Union[str, Model]] = None,
        *,
        memory: Optional[Union[bool, str, Memory]] = None,
        verbose: Optional[INT] = None,
    ) -> Pipeline:
        """Export the pipeline to a sklearn-like object.

        Optionally, you can add a model as final estimator. The
        returned pipeline is already fitted on the training set.

        !!! info
            The returned pipeline behaves similarly to sklearn's
            [Pipeline][], and additionally:

            - Accepts transformers that change the target column.
            - Accepts transformers that drop rows.
            - Accepts transformers that only are fitted on a subset of
              the provided dataset.
            - Always returns pandas objects.
            - Uses transformers that are only applied on the training
              set to fit the pipeline, not to make predictions.

        Parameters
        ----------
        model: str, Model or None, default=None
            Model for which to export the pipeline. If the model used
            [automated feature scaling][], the [Scaler][] is added to
            the pipeline. If None, the pipeline in the current branch
            is exported.

        memory: bool, str, Memory or None, default=None
            Used to cache the fitted transformers of the pipeline.
                - If None or False: No caching is performed.
                - If True: A default temp directory is used.
                - If str: Path to the caching directory.
                - If Memory: Object with the joblib.Memory interface.

        verbose: int or None, default=None
            Verbosity level of the transformers in the pipeline. If
            None, it leaves them to their original verbosity. Note
            that this is not the pipeline's own verbose parameter.
            To change that, use the `set_params` method.

        Returns
        -------
        Pipeline
            Sklearn-like Pipeline object with all transformers in the
            current branch.

        """
        if model:
            model = getattr(self, self._get_models(model)[0])
            pipeline = model.pipeline
        else:
            pipeline = self.pipeline

        if len(pipeline) == 0 and not model:
            raise RuntimeError("There is no pipeline to export!")

        steps = []
        for transformer in pipeline:
            est = deepcopy(transformer)  # Not clone to keep fitted

            # Set the new verbosity (if possible)
            if verbose is not None and hasattr(est, "verbose"):
                est.verbose = verbose

            steps.append((get_pl_name(est.__class__.__name__, steps), est))

        if model:
            steps.append((model.name, deepcopy(model.estimator)))

        if not memory:  # None or False
            memory = None
        elif memory is True:
            memory = tempfile.gettempdir()

        return Pipeline(steps, memory=memory)  # ATOM's pipeline, not sklearn

    @composed(crash, typechecked)
    def get_class_weight(
        self,
        dataset: str = "train",
        target: Union[int, str] = 0,
    ) -> dict:
        """Return class weights for a balanced dataset.

        Statistically, the class weights re-balance the data set so
        that the sampled data set represents the target population
        as closely as possible. The returned weights are inversely
        proportional to the class frequencies in the selected data set.

        Parameters
        ----------
        dataset: str, default="train"
            Data set from which to get the weights. Choose from:
            "train", "test" or "dataset".

        target: int or str, default=0
            Target column to get the class weights from. Only for
            [multioutput tasks][].

        Returns
        -------
        dict
            Classes with the corresponding weights.

        """
        if self.goal != "class":
            raise PermissionError(
                "The get_class_weight method is only available for classification tasks!"
            )

        if dataset not in ("train", "test", "dataset"):
            raise ValueError(
                "Invalid value for the dataset parameter. "
                "Choose from: train, test or dataset."
            )

        y = self.classes[dataset]
        if self._is_multioutput:
            y = y.loc[target if isinstance(target, str) else self.y.columns[target]]

        return {idx: round(divide(sum(y), value), 3) for idx, value in y.items()}

    @composed(crash, method_to_log, typechecked)
    def merge(self, other: Any, /, suffix: str = "2"):
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
            Conflicting branches and models are merged adding `suffix`
            to the end of their names.

        """
        if other.__class__.__name__ != self.__class__.__name__:
            raise TypeError(
                "Invalid class for the other parameter. Expecting a "
                f"{self.__class__.__name__} instance, got {other.__class__.__name__}."
            )

        # Check that both instances have the same original dataset
        current_og = self._get_og_branches()[0]
        other_og = other._get_og_branches()[0]
        if not current_og._data.equals(other_og._data):
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

        self.log("Merging instances...", 1)
        for name, branch in other._branches.min("og").items():
            self.log(f" --> Merging branch {name}.", 1)
            if name in self._branches:
                name = f"{name}{suffix}"
            branch._name = name
            self._branches[name] = branch

        for name, model in other._models.items():
            self.log(f" --> Merging model {name}.", 1)
            if name in self._models:
                name = f"{name}{suffix}"
            model._name = name
            self._models[name] = model

        self.log(" --> Merging attributes.", 1)
        if hasattr(self, "missing"):
            self.missing.extend([x for x in other.missing if x not in self.missing])
        for name, error in other._errors.items():
            if name in self._errors:
                name = f"{name}{suffix}"
            self._errors[name] = error

    @composed(crash, method_to_log, typechecked)
    def stacking(
        self,
        name: str = "Stack",
        models: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Add a [Stacking][] model to the pipeline.

        Parameters
        ----------
        name: str, default="Stack"
            Name of the model. The name is always presided with the
            model's acronym: `stack`.

        models: slice, sequence or None, default=None
            Models that feed the stacking estimator. If None, it selects
            all non-ensemble models trained on the current branch.

        **kwargs
            Additional keyword arguments for sklearn's stacking instance.
            The model's acronyms can be used for the `final_estimator`
            parameter.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_models(models or self.branch._get_depending_models(), False)

        if len(models) < 2:
            raise ValueError(
                "Invalid value for the models parameter. A Stacking model should "
                f"contain at least two underlying estimators, got {models}."
            )

        if not name.lower().startswith("stack"):
            name = f"Stack{name}"

        if name in self._models:
            raise ValueError(
                "Invalid value for the name parameter. It seems a model with "
                f"the name {name} already exists. Add a different name to "
                "train multiple Stacking models within the same instance."
            )

        if isinstance(kwargs.get("final_estimator"), str):
            if kwargs["final_estimator"] not in MODELS:
                raise ValueError(
                    "Invalid value for the final_estimator parameter. Unknown model: "
                    f"{kwargs['final_estimator']}. Choose from: {', '.join(MODELS)}."
                )
            else:
                model = MODELS[kwargs["final_estimator"]](self)
                if self.goal not in model._estimators:
                    raise ValueError(
                        "Invalid value for the final_estimator parameter. Model "
                        f"{model._fullname} can not perform {self.task} tasks."
                    )

                kwargs["final_estimator"] = model._get_est()

        self._models[name] = Stacking(self, name, models=self._models[models], **kwargs)

        if self.experiment:
            self[name]._run = mlflow.start_run(run_name=self[name].name)

        self[name].fit()

        if self.experiment:
            mlflow.end_run()

    @composed(crash, method_to_log, typechecked)
    def voting(
        self,
        name: str = "Vote",
        models: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Add a [Voting][] model to the pipeline.

        Parameters
        ----------
        name: str, default="Vote"
            Name of the model. The name is always presided with the
            model's acronym: `vote`.

        models: slice, sequence or None, default=None
            Models that feed the voting estimator. If None, it selects
            all non-ensemble models trained on the current branch.

        **kwargs
            Additional keyword arguments for sklearn's voting instance.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_models(models or self.branch._get_depending_models(), False)

        if len(models) < 2:
            raise ValueError(
                "Invalid value for the models parameter. A Voting model should "
                f"contain at least two underlying estimators, got {models}."
            )

        if not name.lower().startswith("vote"):
            name = f"Vote{name}"

        if name in self._models:
            raise ValueError(
                "Invalid value for the name parameter. It seems a model with "
                f"the name {name} already exists. Add a different name to "
                "train multiple Voting models within the same instance."
            )

        self._models[name] = Voting(self, name, models=self._models[models], **kwargs)

        if self.experiment:
            self[name]._run = mlflow.start_run(run_name=self[name].name)

        self[name].fit()

        if self.experiment:
            mlflow.end_run()

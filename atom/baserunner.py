# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseRunner class.

"""

import re
from typing import Any, List, Optional, Tuple, Union

import mlflow
import pandas as pd
from typeguard import typechecked

from atom.branch import Branch
from atom.models import MODELS, Stacking, Voting
from atom.utils import (
    DF_ATTRS, FLOAT, INT, SEQUENCE_TYPES, CustomDict, Model, check_is_fitted,
    composed, crash, divide, flt, get_best_score, get_versions, lst,
    method_to_log,
)


class BaseRunner:
    """Base class for runners.

    Contains shared attributes and methods for the atom and trainer
    classes. Implements magic methods, mlflow tracking properties,
    utility properties, prediction methods and utility methods.

    """

    # Tracking parameters for mlflow
    _tracking_params = dict(
        log_bo=True,
        log_model=True,
        log_plots=True,
        log_data=False,
        log_pipeline=False,
    )

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
        if item == "branch":
            self.branch.delete()
        elif item in self._branches:
            self._branches[item].delete()
        elif item == "winner" or item in self._models:
            self.delete(item)
        elif item in self.__dict__:
            del self.__dict__[item]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'."
            )

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
    def log_bo(self) -> bool:
        """Whether to track every trial of the hyperparameter tuning."""
        return self._tracking_params["log_bo"]

    @log_bo.setter
    @typechecked
    def log_bo(self, value: bool):
        self._tracking_params["log_bo"] = value

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
    def branch(self) -> Branch:
        """Return the current branch."""
        return self._branches[self._current]

    @property
    def models(self) -> Union[str, List[str]]:
        """Return the names of all models."""
        if isinstance(self._models, CustomDict):
            return flt([model.name for model in self._models.values()])
        else:
            return self._models

    @property
    def metric(self) -> Union[str, List[str]]:
        """Return the name of the metric."""
        if isinstance(self._metric, CustomDict):
            return flt([metric.name for metric in self._metric.values()])
        else:
            return self._metric

    @property
    def errors(self) -> CustomDict:
        """Return the errors encountered during model training."""
        return self._errors

    @property
    def winners(self) -> List[str]:
        """Return the model names ordered by performance."""
        if self._models:  # Returns None if not fitted
            models = sorted(self._models.values(), key=lambda x: get_best_score(x))
            return [m.name for m in models[::-1]]

    @property
    def winner(self) -> Model:
        """Return the best performing model."""
        if self._models:  # Returns None if not fitted
            return self._models[self.winners[0]]

    @property
    def results(self) -> pd.DataFrame:
        """Return the results as a pd.DataFrame."""

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
                    [[frac(m) for m in self._models.values()], self.models],
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
        index: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        return_test: bool = True,
        branch: Optional[Branch] = None,
    ) -> List[Any]:
        """Get a subset of the rows.

        Parameters
        ----------
        index: int, str, slice, sequence or None, default=None
            Names or indices of the rows to select. If None,
            returns the complete dataset or the test set.

        return_test: bool, default=True
            Whether to return the test or the complete dataset
            when no index is provided.

        branch: Branch or None, default=None
            Get columns from specified branch. If None, use the
            current branch.

        Returns
        -------
        list
            Indices of the included rows.

        """
        if not branch:
            branch = self.branch

        indices = list(branch.dataset.index)
        if branch.holdout is not None:
            indices += list(branch.holdout.index)

        if index is None:
            inc = list(branch._idx[1]) if return_test else list(branch.X.index)
        elif isinstance(index, slice):
            inc = indices[index]
        else:
            inc = []
            for idx in lst(index):
                if idx in indices:
                    inc.append(idx)
                elif isinstance(idx, int):
                    if -len(indices) <= idx <= len(indices):
                        inc.append(indices[idx])
                    else:
                        raise ValueError(
                            f"Invalid value for the index parameter. Value {index} is "
                            f"out of range for a dataset with length {len(indices)}."
                        )
                else:
                    raise ValueError(
                        "Invalid value for the index parameter. "
                        f"Value {idx} not found in the dataset."
                    )

        if not inc:
            raise ValueError(
                "Invalid value for the index parameter, got "
                f"{index}. At least one row has to be selected."
            )

        return inc

    def _get_columns(
        self,
        columns: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        include_target: bool = True,
        return_inc_exc: bool = False,
        only_numerical: bool = False,
        branch: Optional[Branch] = None,
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """Get a subset of the columns.

        Select columns in the dataset by name, index, dtype or regex
        pattern. Duplicate columns are ignored. Exclude columns if
        the string starts with `!`.

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
                else:
                    array = inc
                    if col.startswith("!") and col not in df.columns:
                        array = exc
                        col = col[1:]

                    # Find columns using regex matches
                    matches = [c for c in df.columns if re.fullmatch(col, c)]
                    if matches:
                        array.extend(matches)
                    else:
                        try:
                            array.extend(list(df.select_dtypes(col).columns))
                        except TypeError:
                            raise ValueError(
                                "Invalid value for the columns parameter. "
                                f"Could not find any column that matches {col}."
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
        models: Optional[Union[INT, str, slice, SEQUENCE_TYPES]] = None,
        ensembles: bool = True,
    ) -> List[str]:
        """Get names of models.

        Select models in the dataset by name, index or regex pattern.
        Duplicate models are ignored. Exclude models if the string
        starts with `!`. The input is case-insensitive.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Names or indices of the models to select. If None, it
            returns all models.

        ensembles: bool, default=True
            Whether to include ensemble models in the output. If False,
            they are silently ignored.

        Returns
        -------
        list
            List of model names.

        """
        inc, exc = [], []
        available_models = self._models.min("og")
        if models is None:
            inc.extend(available_models)
        elif isinstance(models, slice):
            inc.extend(available_models[models])
        else:
            for model in lst(models):
                if isinstance(model, int):
                    try:
                        inc.append(available_models[model].name)
                    except KeyError:
                        raise ValueError(
                            "Invalid value for the models parameter. Value "
                            f"{model} is out of range for a pipeline with "
                            f"{len(available_models)} models."
                        )
                else:
                    array = inc
                    if model.startswith("!"):
                        array = exc
                        model = model[1:]

                    if model.lower() == "winner":
                        array.append(self.winner.name)
                    else:
                        # Find models using regex matches
                        matches = [
                            v.name for k, v in available_models.items()
                            if re.fullmatch(model.lower(), k.lower())
                        ]
                        if matches:
                            array.extend(matches)
                        else:
                            raise ValueError(
                                "Invalid value for the models parameter. Could "
                                f"not find any model that matches {model}. The "
                                f"available models are: {', '.join(available_models)}."
                            )

        if inc and exc:
            raise ValueError(
                "Invalid value for the models parameter. You can either "
                "include or exclude models, not combinations of these."
            )

        if exc:
            # If models were excluded with `!`, select all but those
            inc = [m for m in available_models if m not in exc]

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
            - **fullname:** Complete name of the model.
            - **estimator:** The model's underlying estimator.
            - **module:** The estimator's module.
            - **needs_scaling:** Whether the model requires feature scaling.
            - **accepts_sparse:** Whether the model accepts sparse matrices.
            - **supports_engines:** List of engines supported by the model.

        """
        rows = []
        for model in MODELS.values():
            m = model(self, fast_init=True)
            if self.goal in m._estimators:
                rows.append(
                    {
                        "acronym": m.acronym,
                        "fullname": m.fullname,
                        "estimator": m.est_class.__name__,
                        "module": m.est_class.__module__.split(".")[0] + m._module,
                        "needs_scaling": str(m.needs_scaling),
                        "accepts_sparse": str(m.accepts_sparse),
                        "supports_engines": ", ". join(m.supports_engines),
                    }
                )

        return pd.DataFrame(rows)

    @composed(crash, method_to_log)
    def clear(self):
        """Clear attributes from all models.

        Reset all model attributes to their initial state, deleting
        potentially large data arrays. Use this method to free some
        memory before [saving][self-save] the instance. The cleared
        attributes per model are:

        - [Prediction attributes][]
        - [Metric scores][metric]
        - [Shap values][shap]
        - [App instance][adaboost-create_app]
        - [Dashboard instance][adaboost-create_dashboard]

        """
        for model in self._models.values():
            model.clear()

    @composed(crash, method_to_log, typechecked)
    def delete(self, models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None):
        """Delete models.

        If all models are removed, the metric is reset. Use this method
        to drop unwanted models from the pipeline or to free some memory
        before [saving][self-save]. Deleted models are not removed from
        any active [mlflow experiment][tracking].

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Names, positions or regex pattern of the models to delete.
            If None, all models are deleted.

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
    def get_class_weight(self, dataset: str = "train") -> dict:
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

        Returns
        -------
        dict
            Classes with the corresponding weights.

        """
        if self.goal != "class":
            raise PermissionError(
                "The balance method is only available for classification tasks!"
            )

        if dataset not in ("train", "test", "dataset"):
            raise ValueError(
                "Invalid value for the dataset parameter. "
                "Choose from: train, test or dataset."
            )

        y = self.classes[dataset]
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
            branch.name = name
            self._branches[name] = branch

        for name, model in other._models.items():
            self.log(f" --> Merging model {name}.", 1)
            if name in self._models:
                name = f"{name}{suffix}"
            model.name = name
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
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Add a [Stacking][] model to the pipeline.

        Parameters
        ----------
        name: str, default="Stack"
            Name of the model. The name is always presided with the
            model's acronym: `stack`.

        models: sequence or None, default=None
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
                        f"{model.fullname} can not perform {self.task} tasks."
                    )

                kwargs["final_estimator"] = model.get_estimator()

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
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Add a [Voting][] model to the pipeline.

        Parameters
        ----------
        name: str, default="Vote"
            Name of the model. The name is always presided with the
            model's acronym: `vote`.

        models: sequence or None, default=None
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

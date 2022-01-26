# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BasePredictor class.

"""

# Standard packages
import mlflow
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Union, Optional, Any

# Own modules
from .branch import Branch
from .models import MODELS, Stacking, Voting
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, DF_ATTRS, flt, lst,
    check_is_fitted, divide, get_best_score, delete,
    method_to_log, composed, crash, CustomDict,
)


class BasePredictor:
    """Properties and shared methods for the trainers."""

    # Tracking parameters for mlflow
    _tracking_params = dict(
        log_bo=True,
        log_model=True,
        log_plots=True,
        log_data=False,
        log_pipeline=False,
    )

    def __getattr__(self, item):
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
                f"{self.__class__.__name__} object has no attribute {item}."
            )

    def __setattr__(self, item, value):
        if item != "holdout" and isinstance(getattr(Branch, item, None), property):
            setattr(self.branch, item, value)
        else:
            super().__setattr__(item, value)

    def __delattr__(self, item):
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

    def __len__(self):
        return len(self.dataset)

    def __contains__(self, item):
        if self.dataset is None:
            return False
        else:
            return item in self.dataset

    def __getitem__(self, item):
        if self.dataset is None:
            raise RuntimeError(
                "This instance has no dataset annexed to it. Use the "
                "run() method before getting an item from the trainer."
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
    def log_bo(self):
        return self._tracking_params["log_bo"]

    @log_bo.setter
    @typechecked
    def log_bo(self, value: bool):
        self._tracking_params["log_bo"] = value

    @property
    def log_model(self):
        return self._tracking_params["log_model"]

    @log_model.setter
    @typechecked
    def log_model(self, value: bool):
        self._tracking_params["log_model"] = value

    @property
    def log_plots(self):
        return self._tracking_params["log_plots"]

    @log_plots.setter
    @typechecked
    def log_plots(self, value: bool):
        self._tracking_params["log_plots"] = value

    @property
    def log_data(self):
        return self._tracking_params["log_data"]

    @log_data.setter
    @typechecked
    def log_data(self, value: bool):
        self._tracking_params["log_data"] = value

    @property
    def log_pipeline(self):
        return self._tracking_params["log_pipeline"]

    @log_pipeline.setter
    @typechecked
    def log_pipeline(self, value: bool):
        self._tracking_params["log_pipeline"] = value

    # Utility properties =========================================== >>

    @property
    def branch(self):
        """Return the current branch."""
        return self._branches[self._current]

    @property
    def models(self):
        """Return the names of the models in the pipeline."""
        if isinstance(self._models, CustomDict):
            return flt([model.name for model in self._models.values()])
        else:
            return self._models

    @property
    def metric(self):
        """Return the names of the metrics in the pipeline."""
        if isinstance(self._metric, CustomDict):
            return flt([metric.name for metric in self._metric.values()])
        else:
            return self._metric

    @property
    def errors(self):
        """Return the errors encountered during model training."""
        return self._errors

    @property
    def winner(self):
        """Return the best performing model."""
        if self._models:  # Returns None if not fitted
            best_index = np.argmax([get_best_score(m) for m in self._models.values()])
            return self._models[best_index]  # CustomDict can select item from index

    @property
    def results(self):
        """Return the results as a pd.DataFrame."""

        def frac(m):
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

    # Prediction methods =========================================== >>

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Get the winning model's predictions on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.predict(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Get the winning model's probability predictions on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.predict_proba(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Get the winning model's log probability predictions on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.predict_log_proba(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def decision_function(self, X: X_TYPES, **kwargs):
        """Get the winning model's decision function on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.decision_function(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: X_TYPES,
        y: Y_TYPES,
        metric: Optional[Union[str, callable]] = None,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        **kwargs,
    ):
        """Get the winning model's score on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.score(X, y, metric, sample_weight, **kwargs)

    # Utility methods ============================================== >>

    def _get_og_branches(self):
        """Return branches containing the original dataset."""
        return [branch for branch in self._branches.values() if branch.pipeline.empty]

    def _get_rows(self, index=None, return_test=True, branch=None):
        """Get a subset of the rows.

        Parameters
        ----------
        index: int, str, slice, sequence or None, optional (default=None)
            Names or indices of the rows to select. If None,
            returns the complete dataset or the test set.

        return_test: bool, optional (default=True)
            Whether to return the test or the complete dataset
            when no index is provided.

        branch: Branch or None, optional (default=None)
            Get columns from specified branch. If None, use the
            current branch.

        Returns
        -------
        inc: list
            Indices of the included rows.

        """
        if not branch:
            branch = self.branch

        indices = list(branch.dataset.index)
        if branch.holdout is not None:
            indices += list(branch.holdout.index)

        if index is None:
            inc = list(branch.idx[1]) if return_test else list(branch.X.index)
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
        columns=None,
        include_target=True,
        return_inc_exc=False,
        only_numerical=False,
        branch=None,
    ):
        """Get a subset of the columns.

        Select columns in the dataset by name, index or dtype. Duplicate
        columns are ignored. Exclude columns if their name start with `!`.

        Parameters
        ----------
        columns: int, str, slice, sequence or None
            Names, indices or dtypes of the columns to get. If None,
            it returns all columns in the dataframe.

        include_target: bool, optional (default=True)
            Whether to include the target column in the dataframe to
            select from.

        return_inc_exc: bool, optional (default=False)
            Whether to return only included columns or the tuple
            (included, excluded).

        only_numerical: bool, optional (default=False)
            Whether to return only numerical columns.

        branch: Branch or None, optional (default=None)
            Get columns from specified branch. If None, use the current
            branch.

        Returns
        -------
        inc: list
            Names of the included columns.

        exc: list
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
                            f"Invalid value for the columns parameter, got {col} "
                            f"but length of columns is {len(df.columns)}."
                        )
                else:
                    if col not in df.columns:
                        if col.startswith("!"):
                            col = col[1:]
                            if col in df.columns:
                                exc.append(col)
                            else:
                                try:
                                    exc.extend(list(df.select_dtypes(col).columns))
                                except TypeError:
                                    raise ValueError(
                                        "Invalid value for the columns parameter. "
                                        f"Column {col} not found in the dataset."
                                    )
                        else:
                            try:
                                inc.extend(list(df.select_dtypes(col).columns))
                            except TypeError:
                                raise ValueError(
                                    "Invalid value for the columns parameter. "
                                    f"Column {col} not found in the dataset."
                                )
                    else:
                        inc.append(col)

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

    def _get_model_name(self, model):
        """Return a model's name.

        If there are multiple models that start with the same
        acronym, all are returned. If the input is a number,
        select all models that end with that number. The input
        is case-insensitive.

        """
        if model.lower() == "winner":
            return [self.winner.name]

        if model in self._models:
            return [self._models[model].name]

        to_return = []
        for key, value in self._models.items():
            key, model = key.lower(), model.lower()
            condition_1 = key.startswith(model)
            condition_2 = model.replace(".", "").isdigit() and key.endswith(model)
            if condition_1 or condition_2:
                to_return.append(value.name)

        if to_return:
            return to_return
        else:
            raise ValueError(
                f"Model {model} not found in the pipeline! The "
                f"available models are: {', '.join(self._models)}."
            )

    def _get_models(self, models=None, ensembles=True):
        """Return models in the pipeline."""
        if not models:
            if self._models:
                to_return = lst(self.models).copy()
            else:
                to_return = []
        elif isinstance(models, str):
            to_return = self._get_model_name(models.lower())
        else:
            to_return = []
            for m1 in models:
                for m2 in self._get_model_name(m1.lower()):
                    to_return.append(m2)

        if not ensembles:
            to_return = [
                model for model in to_return
                if not model.startswith("Stack") and not model.startswith("Vote")
            ]

        return list(dict.fromkeys(to_return))  # Avoid duplicates

    @crash
    def available_models(self):
        """Give an overview of the available predefined models.

        Returns
        -------
        overview: pd.DataFrame
            Information about the predefined models available for the
            current task. Columns include:
                - acronym: Model's acronym (used to call the model).
                - fullname: Complete name of the model.
                - estimator: The model's underlying estimator.
                - module: The estimator's module.
                - needs_scaling: Whether the model requires feature scaling.
                - accepts_sparse: Whether the model supports sparse matrices.

        """
        overview = pd.DataFrame()
        for model in MODELS.values():
            m = model(self, fast_init=True)
            if self.goal in m.goal:
                overview = overview.append(
                    {
                        "acronym": m.acronym,
                        "fullname": m.fullname,
                        "estimator": m.est_class.__name__,
                        "module": m.est_class.__module__,
                        "needs_scaling": str(m.needs_scaling),
                        "accepts_sparse": str(m.accepts_sparse),
                    },
                    ignore_index=True,
                )

        return overview

    @composed(crash, method_to_log)
    def clear(self):
        """Clear attributes from all models.

        Reset attributes to their initial state, deleting potentially
        large data arrays. Use this method to free some memory before
        saving the class. The cleared attributes per model are:
            - Prediction attributes.
            - Metrics scores.
            - Shap values.

        """
        for model in self._models.values():
            model.clear()

    @composed(crash, method_to_log, typechecked)
    def delete(self, models: Optional[Union[str, SEQUENCE_TYPES]] = None):
        """Delete models from the trainer.

        If all models are removed, the metric is reset. Use this
        method to drop unwanted models from the pipeline or to free
        some memory before saving. Deleted models are not removed
        from any active mlflow experiment.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Models to delete. If None, delete them all.

        """
        models = self._get_models(models)
        if not models:
            self.log(f"No models to delete.", 1)
        else:
            delete(self, models)
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
        threshold: float = 0.5,
    ):
        """Get all models' scores for the provided metrics.

        Parameters
        ----------
        metric: str, func, scorer, sequence or None, optional (default=None)
            Metric to calculate. If None, it returns an overview of
            the most common metrics per task.

        threshold: float, optional (default=0.5)
            Threshold between 0 and 1 to convert predicted probabilities
            to class labels. Only used when:
                - The task is binary classification.
                - The model has a `predict_proba` method.
                - The metric evaluates predicted target values.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

        Returns
        -------
        scores: pd.DataFrame
            Scores of the models.

        """
        check_is_fitted(self, attributes="_models")

        scores = pd.DataFrame()
        for m in self._models.values():
            scores = scores.append(m.evaluate(metric, dataset, threshold))

        return scores

    @composed(crash, typechecked)
    def get_class_weight(self, dataset: str = "train"):
        """Return class weights for a balanced dataset.

        Statistically, the class weights re-balance the data set so
        that the sampled data set represents the target population
        as closely as possible. The returned weights are inversely
        proportional to the class frequencies in the selected data set.

        Parameters
        ----------
        dataset: str, optional (default="train")
            Data set from which to get the weights. Choose from:
            "train", "test" or "dataset".

        Returns
        -------
        class_weights: dict
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
    def merge(self, other: Any, suffix: str = "2"):
        """Merge another trainer into this one.

        Branches, models, metrics and attributes of the other trainer
        are merged into this one. If there are branches and/or models
        with the same name, they are merged adding the `suffix`
        parameter to their name. The errors and missing attributes are
        extended with those of the other instance. It's only possible
        to merge two instances if they are initialized with the same
        dataset and trained with the same metric.

        Parameters
        ----------
        other: trainer
            Trainer instance with which to merge.

        suffix: str, optional (default="2")
            Conflicting branches and models are merged adding `suffix`
            to the end of their names.

        """
        if not hasattr(other, "_branches"):
            raise TypeError(
                "Invalid type for the other parameter. Expecting a "
                f"trainer instance, got {other.__class__.__name__}."
            )

        # Check that both instances have the same original dataset
        if not self._get_og_branches()[0].data.equals(other._get_og_branches()[0].data):
            raise ValueError(
                "Invalid value for the other parameter. The provided trainer "
                "was initialized with a different dataset than this one."
            )

        # Check that both instances have the same metric
        if not self._metric:
            self._metric = other._metric
        elif other.metric and self.metric != other.metric:
            raise ValueError(
                "Invalid value for the other parameter. The provided trainer uses "
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
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Add a Stacking model to the pipeline.

        Parameters
        ----------
        name: str, optional (default="Stack")
            Name of the model. The name is always presided with the
            model's acronym: `stack`.

        models: sequence or None, optional (default=None)
            Models that feed the voting estimator. If None, it selects
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

        if isinstance(kwargs.get("final_estimator"), str):
            if kwargs["final_estimator"] not in MODELS:
                raise ValueError(
                    "Invalid value for the final_estimator parameter. Unknown model: "
                    f"{kwargs['final_estimator']}. Choose from: {', '.join(MODELS)}."
                )
            else:
                model = MODELS[kwargs["final_estimator"]](self)
                if self.goal not in model.goal:
                    raise ValueError(
                        "Invalid value for the final_estimator parameter. Model "
                        f"{model.fullname} can not perform {self.task} tasks."
                    )

                kwargs["final_estimator"] = model.get_estimator()

        self._models[name] = Stacking(self, name, models=self._models[models], **kwargs)

        if self.experiment:
            self[name]._run = mlflow.start_run(run_name=self[name].name)

        self[name].fit()

    @composed(crash, method_to_log, typechecked)
    def voting(
        self,
        name: str = "Vote",
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Add a Voting model to the pipeline.

        Parameters
        ----------
        name: str, optional (default="Vote")
            Name of the model. The name is always presided with the
            model's acronym: `vote`.

        models: sequence or None, optional (default=None)
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

        self._models[name] = Voting(self, name, models=self._models[models], **kwargs)

        if self.experiment:
            self[name]._run = mlflow.start_run(run_name=self[name].name)

        self[name].fit()

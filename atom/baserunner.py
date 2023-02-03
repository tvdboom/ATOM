# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseRunner class.

"""

from __future__ import annotations

import re
from typing import Any, Callable

import mlflow
from joblib.memory import Memory
from sklearn.base import clone
from sklearn.multioutput import (
    ClassifierChain, MultiOutputClassifier, MultiOutputRegressor,
    RegressorChain,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.metaestimators import available_if
from typeguard import typechecked

from atom.basemodel import BaseModel
from atom.basetracker import BaseTracker
from atom.basetransformer import BaseTransformer
from atom.branch import Branch
from atom.models import MODELS, Stacking, Voting
from atom.pipeline import Pipeline
from atom.utils import (
    DF_ATTRS, FLOAT, INT, INT_TYPES, SEQUENCE, SERIES, ClassMap, CustomDict,
    Model, Predictor, bk, check_is_fitted, composed, crash, divide,
    export_pipeline, flt, get_best_score, get_versions, has_task,
    is_multioutput, lst, method_to_log, pd,
)


class BaseRunner(BaseTracker):
    """Base class for runners.

    Contains shared attributes and methods for the atom and trainer
    classes. Implements magic methods, mlflow tracking properties,
    utility properties, prediction methods and utility methods.

    """

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
        if item in self.__dict__.get("_branches"):
            return self._branches[item]  # Get branch
        elif item in dir(self.branch) and not item.startswith("_"):
            return getattr(self.branch, item)  # Get attr from branch
        elif item in self.__dict__.get("_models"):
            return self._models[item]  # Get model
        elif item in self.branch.columns:
            return self.branch.dataset[item]  # Get column from dataset
        elif item in DF_ATTRS:
            return getattr(self.branch.dataset, item)  # Get attr from dataset
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'."
            )

    def __setattr__(self, item: str, value: Any):
        if isinstance(getattr(Branch, item, None), property):
            setattr(self.branch, item, value)
        else:
            super().__setattr__(item, value)

    def __delattr__(self, item: str):
        if item in self._models:
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

    def __getitem__(self, item: INT | str | list) -> Any:
        if self.dataset is None:
            raise RuntimeError(
                "This instance has no dataset annexed to it. "
                "Use the run method before calling __getitem__."
            )
        elif isinstance(item, INT_TYPES):
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
        elif isinstance(item, list):
            return self.dataset[item]  # Get subset of dataset
        else:
            raise TypeError(
                f"{self.__class__.__name__} is only "
                "subscriptable with types int, str or list."
            )

    # Utility properties =========================================== >>

    @property
    def og(self) -> Branch:
        """Branch containing the original dataset.

        This branch contains the data prior to any transformations.
        It redirects to the current branch if its pipeline is empty
        to not have the same data in memory twice.

        """
        return self._og or self.branch

    @property
    def branch(self) -> Branch:
        """Current active branch.

        Use the property's `@setter` to change the branch or to create
        a new one. If the value is the name of an existing branch,
        switch to that one. Else, create a new branch using that name.
        The new branch is split from the current branch. Use `__from__`
        to split the new branch from any other existing branch. Read
        more in the [user guide][branches].

        """
        return self._current

    @branch.deleter
    def branch(self):
        """Delete the current active branch."""
        if len(self._branches) == 1:
            raise PermissionError("Can't delete the last branch!")

        # Delete all depending models
        for model in self._models:
            if model.branch is self.branch:
                self._delete_models(model.name)

        self._branches.remove(self._current)
        self._current = self._branches[-1]

        self.log(f"Branch {self.branch.name} successfully deleted.", 1)
        self.log(f"Switched to branch {self.branch.name}.", 1)

    @property
    def multioutput(self) -> Predictor | None:
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
    def multioutput(self, value: str | Predictor | None):
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
    def models(self) -> str | list[str] | None:
        """Name of the model(s)."""
        if isinstance(self._models, ClassMap):
            return flt(self._models.keys())

    @property
    def metric(self) -> str | list[str] | None:
        """Name of the metric(s)."""
        if isinstance(self._metric, ClassMap):
            return flt(self._metric.keys())

    @property
    def winners(self) -> list[Model]:
        """Models ordered by performance.

        Performance is measured as the highest score on the model's
        [`score_bootstrap`][adaboost-score_bootstrap] or
        [`score_test`][adaboost-score_test] attributes, checked in
        that order. For [multi-metric runs][], only the main metric
        is compared.

        """
        if self._models:  # Returns None if not fitted
            return sorted(self._models, key=lambda x: get_best_score(x), reverse=True)

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

        def frac(m: Model) -> FLOAT:
            """Return the fraction of the train set used for the model."""
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

    def _get_models(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        ensembles: bool = True,
        branch: Branch | None = None,
    ) -> list[Model]:
        """Get models.

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
            they are silently excluded from any return.

        branch: Branch or None, default=None
            Force returned models to have been fitted on this branch,
            else raises an exception. If None, this filter is ignored.

        Returns
        -------
        list
            Selected models.

        """

        def get_match(model: str):
            """Try to find a match by regex.

            Parameters
            ----------
            model: str
                Regex pattern to match with model names.

            """
            nonlocal inc, exc

            array = inc
            if model.startswith("!") and model not in self._models:
                array = exc
                model = model[1:]

            # Find rows using regex matches
            if model.lower() == "winner":
                array.append(self.winner)
            elif match := [m for m in self._models if re.fullmatch(model, m.name, re.I)]:
                array.extend(match)
            else:
                raise ValueError(
                    "Invalid value for the models parameter. Could "
                    f"not find any model that matches {model}. The "
                    f"available models are: {', '.join(self._models.keys())}."
                )

        inc, exc = [], []
        if models is None:
            inc.extend(self._models)
        elif isinstance(models, slice):
            inc.extend(self._models[models])
        else:
            for model in lst(models):
                if isinstance(model, INT_TYPES):
                    try:
                        inc.append(self._models[model])
                    except KeyError:
                        raise ValueError(
                            "Invalid value for the models parameter. Value "
                            f"{model} is out of range for a pipeline with "
                            f"{len(self._models)} models."
                        )
                elif isinstance(model, str):
                    try:
                        get_match(model)
                    except ValueError:
                        for m in model.split("+"):
                            get_match(m)
                elif isinstance(model, BaseModel):
                    inc.append(model)
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
            inc = [m for m in self._models if m not in exc]

        if not ensembles:
            inc = [m for m in inc if all(not m.acronym == x for x in ("Stack", "Vote"))]

        if branch:
            for m in inc:
                if m.branch is not branch:
                    raise ValueError(
                        "Invalid value for the models parameter. All "
                        f"models must have been fitted on {branch}, but "
                        f"model {m.name} is fitted on {m.branch}."
                    )

        return list(dict.fromkeys(inc))  # Avoid duplicates

    def _delete_models(self, models: str | SEQUENCE):
        """Delete models.

        Remove models from the instance. All attributes are deleted
        except for `errors`. If all models are removed, the metric is
        reset.

        Parameters
        ----------
        models: str or sequence
            Model(s) to delete.

        """
        for model in lst(models):
            if model in self._models:
                self._models.remove(model)

        # If no models, reset the metric
        if not self._models:
            self._metric = ClassMap()

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
        for model in MODELS:
            m = model(goal=self.goal)
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
        for model in self._models:
            model.clear()

    @composed(crash, method_to_log, typechecked)
    def delete(
        self,
        models: INT | str | slice | Model | SEQUENCE | None = None
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
            self.log(f"Deleting {len(models)} models...", 1)
            for m in models:
                self._delete_models(m.name)
                self.log(f" --> Model {m.name} successfully deleted.", 1)

    @composed(crash, typechecked)
    def evaluate(
        self,
        metric: str | Callable | SEQUENCE | None = None,
        dataset: str = "test",
        *,
        threshold: FLOAT = 0.5,
        sample_weight: SEQUENCE | None = None,
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

        evaluations = []
        for m in self._models:
            evaluations.append(
                m.evaluate(
                    metric=metric,
                    dataset=dataset,
                    threshold=threshold,
                    sample_weight=sample_weight,
                )
            )

        return pd.DataFrame(evaluations)

    @composed(crash, typechecked)
    def export_pipeline(
        self,
        model: str | Model | None = None,
        *,
        memory: bool | str | Memory | None = None,
        verbose: INT | None = None,
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
            Current branch as a sklearn-like Pipeline object.

        """
        if model:
            model = self._get_models(model)[0]
            pipeline = model.pipeline
        else:
            pipeline = self.pipeline

        if len(pipeline) == 0 and not model:
            raise RuntimeError("There is no pipeline to export!")

        return export_pipeline(pipeline, model, memory, verbose)

    @available_if(has_task("class"))
    @composed(crash, typechecked)
    def get_class_weight(self, dataset: str = "train") -> CustomDict:
        """Return class weights for a balanced data set.

        Statistically, the class weights re-balance the data set so
        that the sampled data set represents the target population
        as closely as possible. The returned weights are inversely
        proportional to the class frequencies in the selected data set.

        Parameters
        ----------
        dataset: str, default="train"
            Data set from which to get the weights. Choose from:
            "train", "test", "dataset".

        Returns
        -------
        dict
            Classes with the corresponding weights. A dict of dicts is
            returned for [multioutput tasks][].

        """
        if dataset.lower() not in ("train", "test", "dataset"):
            raise ValueError(
                f"Invalid value for the dataset parameter, got {dataset}. "
                "Choose from: train, test, dataset."
            )

        y = self.classes[dataset.lower()]
        if is_multioutput(self.task):
            weights = {
                t: {i: round(divide(sum(y.loc[t]), v), 3) for i, v in y.items()}
                for t in self.target
            }
        else:
            weights = {idx: round(divide(sum(y), value), 3) for idx, value in y.items()}

        return CustomDict(weights)

    @available_if(has_task("class"))
    @composed(crash, typechecked)
    def get_sample_weight(self, dataset: str = "train") -> SERIES:
        """Return sample weights for a balanced data set.

        The returned weights are inversely proportional to the class
        frequencies in the selected data set. For [multioutput tasks][],
        the weights of each column of `y` will be multiplied.

        Parameters
        ----------
        dataset: str, default="train"
            Data set from which to get the weights. Choose from:
            "train", "test", "dataset".

        Returns
        -------
        series
            Sequence of weights with shape=(n_samples,).

        """
        if dataset.lower() not in ("train", "test", "dataset"):
            raise ValueError(
                f"Invalid value for the dataset parameter, got {dataset}. "
                "Choose from: train, test, dataset."
            )

        weights = compute_sample_weight("balanced", y=getattr(self, dataset.lower()))
        return bk.Series(weights, name="sample_weight").round(3)

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
        if not self.og._data.equals(other.og._data):
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
        for branch in other._branches:
            self.log(f" --> Merging branch {branch.name}.", 1)
            if branch.name in self._branches:
                branch._name = f"{branch.name}{suffix}"
            self._branches[branch.name] = branch

        for model in other._models:
            self.log(f" --> Merging model {model.name}.", 1)
            if model.name in self._models:
                model._name = f"{model.name}{suffix}"
            self._models[model.name] = model

        self.log(" --> Merging attributes.", 1)
        if hasattr(self, "missing"):
            self.missing.extend([x for x in other.missing if x not in self.missing])

    @composed(crash, method_to_log, typechecked)
    def stacking(
        self,
        name: str = "Stack",
        models: slice | SEQUENCE | None = None,
        **kwargs,
    ):
        """Add a [Stacking][] model to the pipeline.

        !!! warning
            Combining models trained on different branches into one
            ensemble is not allowed and will raise an exception.

        Parameters
        ----------
        name: str, default="Stack"
            Name of the model. The name is always presided with the
            model's acronym: `Stack`.

        models: slice, sequence or None, default=None
            Models that feed the stacking estimator. If None, it selects
            all non-ensemble models trained on the current branch.

        **kwargs
            Additional keyword arguments for sklearn's stacking instance.
            The model's acronyms can be used for the `final_estimator`
            parameter.

        """
        check_is_fitted(self, attributes="_models")
        models = ClassMap(*self._get_models(models, ensembles=False, branch=self.branch))

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

        kw_model = dict(
            index=self.index,
            goal=self.goal,
            metric=self._metric,
            multioutput=self.multioutput,
            og=self.og,
            branch=self.branch,
            **{attr: getattr(self, attr) for attr in BaseTransformer.attrs},
        )

        if isinstance(kwargs.get("final_estimator"), str):
            if kwargs["final_estimator"] not in MODELS:
                raise ValueError(
                    "Invalid value for the final_estimator parameter. "
                    f"Unknown model: {kwargs['final_estimator']}. Choose "
                    f"from: {', '.join(MODELS.keys())}."
                )
            else:
                model = MODELS[kwargs["final_estimator"]](**kw_model)
                if self.goal not in model._estimators:
                    raise ValueError(
                        "Invalid value for the final_estimator parameter. Model "
                        f"{model._fullname} can not perform {self.task} tasks."
                    )

                kwargs["final_estimator"] = model._get_est()

        self._models.append(Stacking(name=name, models=models, **kw_model, **kwargs))

        if self.experiment:
            self[name]._run = mlflow.start_run(run_name=self[name].name)

        self[name].fit()

    @composed(crash, method_to_log, typechecked)
    def voting(
        self,
        name: str = "Vote",
        models: slice | SEQUENCE | None = None,
        **kwargs,
    ):
        """Add a [Voting][] model to the pipeline.

        !!! warning
            Combining models trained on different branches into one
            ensemble is not allowed and will raise an exception.

        Parameters
        ----------
        name: str, default="Vote"
            Name of the model. The name is always presided with the
            model's acronym: `Vote`.

        models: slice, sequence or None, default=None
            Models that feed the voting estimator. If None, it selects
            all non-ensemble models trained on the current branch.

        **kwargs
            Additional keyword arguments for sklearn's voting instance.

        """
        check_is_fitted(self, attributes="_models")
        models = ClassMap(*self._get_models(models, ensembles=False, branch=self.branch))

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

        self._models.append(
            Voting(
                name=name,
                models=models,
                index=self.index,
                goal=self.goal,
                metric=self._metric,
                multioutput=self.multioutput,
                og=self.og,
                branch=self.branch,
                **{attr: getattr(self, attr) for attr in BaseTransformer.attrs},
                **kwargs,
            )
        )

        if self.experiment:
            self[name]._run = mlflow.start_run(run_name=self[name].name)

        self[name].fit()

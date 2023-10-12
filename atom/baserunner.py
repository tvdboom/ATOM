# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the BaseRunner class.

"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import dill as pickle
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Literal
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.metaestimators import available_if

from atom.basetracker import BaseTracker
from atom.basetransformer import BaseTransformer
from atom.branch import Branch
from atom.models import MODELS, Stacking, Voting
from atom.pipeline import Pipeline
from atom.utils.constants import DF_ATTRS
from atom.utils.types import (
    Bool, DataFrame, Float, Int, IntTypes, MetricConstructor, Model,
    ModelSelector, ModelsSelector, Scalar, SegmentTypes, Sequence, Series,
)
from atom.utils.utils import (
    ClassMap, CustomDict, Task, check_is_fitted, composed, crash, divide, flt,
    get_segment, get_versions, has_task, lst, method_to_log,
)


class BaseRunner(BaseTracker):
    """Base class for runners.

    Contains shared attributes and methods for atom and trainers.

    """

    def __getstate__(self) -> dict[str, Any]:
        # Store an extra attribute with the package versions
        return {**self.__dict__, "_versions": get_versions(self._models)}

    def __setstate__(self, state: dict[str, Any]):
        versions = state.pop("_versions", None)
        self.__dict__.update(state)

        # Check that all package versions match or raise a warning
        if versions:
            for key, value in get_versions(state["_models"]).items():
                if versions[key] != value:
                    self._log(
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

    def __contains__(self, item: str) -> Bool:
        return item in self.dataset

    def __getitem__(self, item: Int | str | list) -> Any:
        if self.dataset.empty:
            raise RuntimeError(
                "This instance has no dataset annexed to it. "
                "Use the run method before calling __getitem__."
            )
        elif isinstance(item, IntTypes):
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

    # Utility properties =========================================== >>

    @property
    def task(self) -> Task:
        """Dataset's [task][] type."""
        return self._goal.infer_task(self.y)

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
    def holdout(self) -> DataFrame | None:
        """Holdout set.

        This data set is untransformed by the pipeline. Read more in
        the [user guide][data-sets].

        """
        return self.branch._holdout

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
    def winners(self) -> list[Model] | None:
        """Models ordered by performance.

        Performance is measured as the highest score on the model's
        [`score_bootstrap`][adaboost-score_bootstrap] or
        [`score_test`][adaboost-score_test] attributes, checked in
        that order. For [multi-metric runs][], only the main metric
        is compared. Ties are resolved looking at the lowest
        [time_fit][adaboost-time_fit].

        """
        if self._models:  # Returns None if not fitted
            return sorted(
                self._models, key=lambda x: (x._best_score(), x.time_fit), reverse=True
            )

    @property
    def winner(self) -> Model | None:
        """Best performing model.

        Performance is measured as the highest score on the model's
        [`score_bootstrap`][adaboost-score_bootstrap] or
        [`score_test`][adaboost-score_test] attributes, checked in
        that order. For [multi-metric runs][], only the main metric
        is compared. Ties are resolved looking at the lowest
        [time_fit][adaboost-time_fit].

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

        def frac(m: Model) -> Float:
            """Return the fraction of the train set used.

            Parameters
            ----------
            m: Model
                Used model.

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

    def _get_models(
        self,
        models: ModelsSelector = None,
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
        inc, exc = [], []
        if models is None:
            return self._models.values()
        elif isinstance(models, SegmentTypes):
            return get_segment(self._models, models)
        else:
            for model in lst(models):
                if isinstance(model, IntTypes):
                    try:
                        inc.append(self._models[model])
                    except KeyError:
                        raise IndexError(
                            f"Invalid value for the models parameter. The {model} is out "
                            f"of range for an instance with {len(self._models)} models."
                        )
                elif isinstance(model, str):
                    for mdl in model.split("+"):
                        array = inc
                        if mdl.startswith("!") and mdl not in self._models:
                            array = exc
                            mdl = mdl[1:]

                        if mdl.lower() == "winner":
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
            inc = list(filter(lambda m: m.acronym not in ("Stack", "Vote"), inc))

        if branch and not all(m.branch is branch for m in inc):
            raise ValueError(
                "Invalid value for the models parameter. Not "
                f"all models have been fitted on {branch}."
            )

        return list(dict.fromkeys(inc))  # Avoid duplicates

    def _delete_models(self, models: str | Sequence):
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
            - **native_multilabel:** Whether the model has native support
              for [multilabel][] tasks.
            - **native_multioutput:** Whether the model has native support
              for [multioutput tasks][].
            - **has_validation:** Whether the model has [in-training validation][].
            - **supports_engines:** Engines supported by the model.

        """
        rows = []
        for model in MODELS:
            m = model(goal=self._goal)
            if self._goal in m._estimators:
                rows.append(
                    {
                        "acronym": m.acronym,
                        "model": m._fullname,
                        "estimator": m._est_class.__name__,
                        "module": m._est_class.__module__.split(".")[0] + m._module,
                        "needs_scaling": m.needs_scaling,
                        "accepts_sparse": m.accepts_sparse,
                        "native_multilabel": m.native_multilabel,
                        "native_multioutput": m.native_multioutput,
                        "has_validation": bool(m.has_validation),
                        "supports_engines": ", ". join(m.supports_engines),
                    }
                )

        return pd.DataFrame(rows)

    @composed(crash, method_to_log)
    def clear(self):
        """Reset attributes and clear memoization from all models.

        Reset certain model attributes to their initial state, deleting
        potentially large data arrays. Use this method to free some
        memory before [saving][self-save] the instance. The affected
        attributes are:

        - [In-training validation][] scores
        - [Shap values][shap]
        - [App instance][adaboost-create_app]
        - [Dashboard instance][adaboost-create_dashboard]
        - Memoized [metric scores][metric]
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
        models = self._get_models(models)
        if not models:
            self._log("No models to delete.", 1)
        else:
            self._log(f"Deleting {len(models)} models...", 1)
            for m in models:
                self._delete_models(m.name)
                self._log(f" --> Model {m.name} successfully deleted.", 1)

    @composed(crash, beartype)
    def evaluate(
        self,
        metric: MetricConstructor = None,
        dataset: Literal["train", "test", "holdout"] = "test",
        *,
        threshold: Float | Sequence[Float] = 0.5,
        sample_weight: Sequence[Scalar] | None = None,
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
            is exported.

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
        dataset: Literal["train", "test", "holdout"] = "train",
    ) -> CustomDict:
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
        y = self.classes[dataset]
        if self.task.is_multioutput:
            weights = {
                t: {i: round(divide(sum(y.loc[t]), v), 3) for i, v in y.items()}
                for t in self.target
            }
        else:
            weights = {idx: round(divide(sum(y), value), 3) for idx, value in y.items()}

        return CustomDict(weights)

    @available_if(has_task("classification"))
    @composed(crash, beartype)
    def get_sample_weight(
        self,
        dataset: Literal["train", "test", "holdout"] = "train",
    ) -> Series:
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
        weights = compute_sample_weight("balanced", y=getattr(self, dataset))
        return pd.Series(weights, name="sample_weight").round(3)

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

        self._log("Merging instances...", 1)
        for branch in other._branches:
            self._log(f" --> Merging branch {branch.name}.", 1)
            if branch.name in self._branches:
                branch._name = f"{branch.name}{suffix}"
            self._branches[branch.name] = branch

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
            og = self._branches.og._container
            self._branches._og._container = None
            for branch in self._branches:
                data[branch.name] = dict(
                    _data=deepcopy(branch._container),
                    _holdout=deepcopy(branch._holdout),
                    holdout=branch.__dict__.pop("holdout", None)  # Clear cached holdout
                )
                branch._container = None
                branch._holdout = None

        if (path := Path(filename)).suffix != ".pkl":
            path = path.with_suffix(".pkl")

        if path.name == "auto.csv":
            path = path.with_name(f"{self.__class__.__name__}.pkl")

        with open(path, "wb") as f:
            pickle.settings["recurse"] = True
            pickle.dump(self, f)

        # Restore the data to the attributes
        if not save_data:
            self._branches._og._container = og
            for branch in self._branches:
                branch._container = data[branch.name]["_data"]
                branch._holdout = data[branch.name]["_holdout"]
                if data[branch.name]["holdout"] is not None:
                    branch.__dict__["holdout"] = data[branch.name]["holdout"]

        self._log(f"{self.__class__.__name__} successfully saved.", 1)

    @composed(crash, method_to_log, beartype)
    def stacking(
        self,
        models: range | slice | Sequence[ModelSelector] | None = None,
        name: str = "Stack",
        **kwargs,
    ):
        """Add a [Stacking][] model to the pipeline.

        !!! warning
            Combining models trained on different branches into one
            ensemble is not allowed and will raise an exception.

        Parameters
        ----------
        models: range, slice, sequence or None, default=None
            Models that feed the stacking estimator. The models must have
            been fitted on the current branch.

        name: str, default="Stack"
            Name of the model. The name is always presided with the
            model's acronym: `Stack`.

        **kwargs
            Additional keyword arguments for sklearn's stacking instance.
            The model's acronyms can be used for the `final_estimator`
            parameter.

        """
        check_is_fitted(self, attributes="_models")
        m = ClassMap(*self._get_models(models, ensembles=False, branch=self.branch))

        if len(m) < 2:
            raise ValueError(
                "Invalid value for the models parameter. A Stacking model should "
                f"contain at least two underlying estimators, got only {m[0]}."
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
            goal=self._goal,
            config=self._config,
            branches=self._branches,
            metric=self._metric,
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
                if self._goal not in model._estimators:
                    raise ValueError(
                        "Invalid value for the final_estimator parameter. Model "
                        f"{model._fullname} can not perform {self.task} tasks."
                    )

                kwargs["final_estimator"] = model._get_est()

        self._models.append(Stacking(models=m, name=name, **kw_model, **kwargs))

        self[name].fit()

    @composed(crash, method_to_log, beartype)
    def voting(
        self,
        models: range | slice | Sequence[ModelSelector] | None = None,
        name: str = "Vote",
        **kwargs,
    ):
        """Add a [Voting][] model to the pipeline.

        !!! warning
            Combining models trained on different branches into one
            ensemble is not allowed and will raise an exception.

        Parameters
        ----------
        models: range, slice, sequence or None, default=None
            Models that feed the stacking estimator. The models must have
            been fitted on the current branch.

        name: str, default="Vote"
            Name of the model. The name is always presided with the
            model's acronym: `Vote`.

        **kwargs
            Additional keyword arguments for sklearn's voting instance.

        """
        check_is_fitted(self, attributes="_models")
        m = ClassMap(*self._get_models(models, ensembles=False, branch=self.branch))

        if len(m) < 2:
            raise ValueError(
                "Invalid value for the models parameter. A Voting model should "
                f"contain at least two underlying estimators, got only {m[0]}."
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
                models=m,
                name=name,
                goal=self._goal,
                config=self._config,
                branches=self._branches,
                metric=self._metric,
                **{attr: getattr(self, attr) for attr in BaseTransformer.attrs},
                **kwargs,
            )
        )

        self[name].fit()

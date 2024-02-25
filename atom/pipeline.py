"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the ATOM's custom sklearn-like pipeline.

"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import clone
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.pipeline import _final_estimator_has
from sklearn.utils import Bunch, _print_elapsed_time
from sklearn.utils._metadata_requests import MetadataRouter, MethodMapping
from sklearn.utils.metadata_routing import _raise_for_params, process_routing
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory
from sktime.forecasting.base import BaseForecaster
from typing_extensions import Self

from atom.utils.types import (
    Bool, EngineDataOptions, EngineTuple, Estimator, FHConstructor, Float,
    Pandas, Scalar, Sequence, Verbose, XConstructor, XReturn, YConstructor,
    YReturn,
)
from atom.utils.utils import (
    NotFittedError, adjust, check_is_fitted, fit_one, fit_transform_one, to_df,
    to_tabular, transform_one, variable_return,
)


if TYPE_CHECKING:
    from sktime.proba.normal import Normal


T = TypeVar("T")


class Pipeline(SkPipeline):
    """Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be transformsers, that
    is, they must implement `fit` and `transform` methods. The final
    estimator only needs to implement `fit`. The transformers in the
    pipeline can be cached using the `memory` parameter.

    A step's estimator may be replaced entirely by setting the
    parameter with its name to another estimator, or a transformer
    removed by setting it to `passthrough` or `None`.

    Read more in sklearn's the [user guide][pipelinedocs].

    !!! info
        This class behaves similarly to sklearn's [pipeline][skpipeline],
        and additionally:

        - Can initialize with an empty pipeline.
        - Always returns 'pandas' objects.
        - Accepts transformers that drop rows.
        - Accepts transformers that only are fitted on a subset of the
          provided dataset.
        - Accepts transformers that apply only on the target column.
        - Uses transformers that are only applied on the training set
          to fit the pipeline, not to make predictions on new data.
        - The instance is considered fitted at initialization if all
          the underlying transformers/estimator in the pipeline are.
        - It returns attributes from the final estimator if they are
          not of the Pipeline.
        - The last estimator is also cached.
        - Supports time series models following sktime's API.

    !!! warning
        This Pipeline only works with estimators whose parameters
        for fit, transform, predict, etc... are named `X` and/or `y`.

    Parameters
    ----------
    steps: list of tuple
        List of (name, transform) tuples (implementing `fit`/`transform`)
        that are chained in sequential order.

    memory: str, [Memory][joblibmemory] or None, default=None
        Used to cache the fitted transformers of the pipeline. Enabling
        caching triggers a clone of the transformers before fitting.
        Therefore, the transformer instance given to the pipeline cannot
        be inspected directly. Use the attribute `named_steps` or `steps`
        to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time-consuming.

    verbose: int or None, default=0
        Verbosity level of the transformers in the pipeline. If None,
        it leaves them to their original verbosity. If >0, the time
        elapsed while fitting each step is printed. Note this is not
        the same as sklearn's `verbose` parameter. Use the pipeline's
        verbose attribute to modify that one (defaults to False).

    Attributes
    ----------
    named_steps: [Bunch][]
        Dictionary-like object, with the following attributes. Read-only
        attribute to access any step parameter by user given name. Keys
        are step names and values are steps parameters.

    classes_: np.ndarray of shape (n_classes,)
        The class' labels. Only exist if the last step of the pipeline
        is a classifier.

    feature_names_in_: np.ndarray
        Names of features seen during first step `fit` method.

    n_features_in_: int
        Number of features seen during first step `fit` method.

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Initialize atom
    atom = ATOMClassifier(X, y, verbose=2)

    # Apply data cleaning and feature engineering methods
    atom.scale()
    atom.balance(strategy="smote")
    atom.feature_selection(strategy="rfe", solver="lr", n_features=22)

    # Train models
    atom.run(models="LR")

    # Get the pipeline object
    pipeline = atom.lr.export_pipeline()
    print(pipeline)
    ```

    """

    def __init__(
        self,
        steps: list[tuple[str, Estimator]],
        *,
        memory: str | Memory | None = None,
        verbose: Verbose | None = 0,
    ):
        super().__init__(steps=steps, memory=memory, verbose=False)
        self._verbose = verbose

    def __bool__(self):
        """Whether the pipeline has at least one estimator."""
        return len(self.steps) > 0

    def __contains__(self, item: str | Any):
        """Whether the name or estimator is in the pipeline."""
        if isinstance(item, str):
            return item in self.named_steps
        else:
            return item in self.named_steps.values()

    def __getattr__(self, item: str):
        """Get the attribute from the final estimator."""
        try:
            return getattr(self._final_estimator, item)
        except (AttributeError, IndexError):
            raise AttributeError(f"'Pipeline' object has no attribute '{item}'.") from None

    def __sklearn_is_fitted__(self):
        """Whether the pipeline has been fitted."""
        try:
            # check if the last step of the pipeline is fitted
            # we only check the last step since if the last step is fit, it
            # means the previous steps should also be fit. This is faster than
            # checking if every step of the pipeline is fit.
            check_is_fitted(self.steps[-1][1])
            return True
        except (NotFittedError, IndexError):
            return False

    @property
    def memory(self) -> Memory:
        """Get the internal memory object."""
        return self._memory

    @memory.setter
    def memory(self, value: str | Memory | None):
        """Create a new internal memory object."""
        self._memory = check_memory(value)
        self._mem_fit = self._memory.cache(fit_one)
        self._mem_fit_transform = self._memory.cache(fit_transform_one)
        self._mem_transform = self._memory.cache(transform_one)

    @property
    def _final_estimator(self) -> Literal["passthrough"] | Estimator | None:
        """Return the last estimator in the pipeline.

        If the pipeline is empty, return None. If the estimator is
        None, return "passthrough".

        """
        try:
            estimator = self.steps[-1][1]
            return "passthrough" if estimator is None else estimator
        except (ValueError, AttributeError, TypeError, IndexError):
            # This condition happens when the pipeline is empty or a call
            # to a method is first calling `_available_if` and `fit` did
            # not validate `steps` yet.
            return None

    def _can_transform(self) -> bool:
        """Check if the pipeline can use the transform method."""
        return (
            self._final_estimator is None
            or self._final_estimator == "passthrough"
            or hasattr(self._final_estimator, "transform")
        )

    def _can_inverse_transform(self) -> bool:
        """Check if the pipeline can use the transform method."""
        return all(
            est is None or est == "passthrough" or hasattr(est, "inverse_transform")
            for _, _, est in self._iter()
        )

    @overload
    def _convert(self, obj: Literal[None]) -> None: ...

    @overload
    def _convert(self, obj: pd.DataFrame) -> XReturn: ...

    @overload
    def _convert(self, obj: pd.Series) -> YReturn: ...

    def _convert(self, obj: Pandas | None) -> YReturn | None:
        """Convert data to the type set in the data engine.

        Parameters
        ----------
        obj: pd.Series, pd.DataFrame or None
            Object to convert. If None, return as is.

        Returns
        -------
        object
            Converted data.

        """
        # Only apply transformations when the engine is defined
        if hasattr(self, "_engine") and isinstance(obj, pd.Series | pd.DataFrame):
            return self._engine.data_engine.convert(obj)
        else:
            return obj

    def _iter(
        self,
        *,
        with_final: Bool = True,
        filter_passthrough: Bool = True,
        filter_train_only: Bool = True,
    ) -> Iterator[tuple[int, str, Estimator]]:
        """Generate (idx, name, estimator) tuples from self.steps.

        By default, estimators that are only applied on the training
        set are filtered out for predictions.

        Parameters
        ----------
        with_final: bool, default=True
            Whether to include the final estimator.

        filter_passthrough: bool, default=True
            Whether to exclude `passthrough` elements.

        filter_train_only: bool, default=True
            Whether to exclude estimators that should only be used for
            training (have `_train_only=True` attribute).

        Yields
        ------
        int
            Index position in the pipeline.

        str
            Name of the estimator.

        Estimator
            Transformer or predictor instance.

        """
        stop = len(self.steps)
        if not with_final and stop > 0:
            stop -= 1

        for idx, (name, trans) in enumerate(islice(self.steps, 0, stop)):
            if (
                (not filter_passthrough or (trans is not None and trans != "passthrough"))
                and (not filter_train_only or not getattr(trans, "_train_only", False))
            ):
                yield idx, name, trans

    def _fit(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        routed_params: dict[str, Bunch] | None = None,
    ) -> tuple[pd.DataFrame | None, Pandas | None]:
        """Get data transformed through the pipeline.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            `X` is ignored. None if the pipeline only uses y.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`.

        routed_params: dict or None, default=None
            Metadata parameters routed for the fit method.

        Returns
        -------
        dataframe or None
            Transformed feature set.

        series, dataframe or None
            Transformed target column.

        """
        self.steps: list[tuple[str, Estimator]] = list(self.steps)
        self._validate_steps()

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        for step, name, transformer in self._iter(
                with_final=False, filter_passthrough=False, filter_train_only=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step)):
                    continue

            # Don't clone when caching is disabled to preserve backward compatibility
            if self.memory.location is None:
                cloned = transformer
            else:
                cloned = clone(transformer)

                # Attach internal attrs otherwise wiped by clone
                for attr in ("_cols", "_train_only"):
                    if hasattr(transformer, attr):
                        setattr(cloned, attr, getattr(transformer, attr))

            with adjust(cloned, verbose=self._verbose):
                # Fit or load the current estimator from cache
                # Type ignore because routed_params is never None but
                # the signature of _fit needs to comply with sklearn's
                Xt, yt, fitted_transformer = self._mem_fit_transform(
                    transformer=cloned,
                    X=Xt,
                    y=yt,
                    message=self._log_message(step),
                    **routed_params[name].fit_transform,  # type: ignore[index]
                )

            # Replace the estimator of the step with the fitted
            # estimator (necessary when loading from cache)
            self.steps[step] = (name, fitted_transformer)

        return Xt, yt

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Check [sklearn's documentation][metadata_routing] on how the
        routing mechanism works.

        Returns
        -------
        MetadataRouter
            A [MetadataRouter][] encapsulating routing information.

        """
        router = MetadataRouter(owner=self.__class__.__name__)

        # First, we add all steps except the last one
        for _, name, trans in self._iter(with_final=False, filter_train_only=False):
            method_mapping = MethodMapping()
            # fit, fit_predict, and fit_transform call fit_transform if it
            # exists, or else fit and transform
            if hasattr(trans, "fit_transform"):
                (
                    method_mapping.add(caller="fit", callee="fit_transform")
                    .add(caller="fit_transform", callee="fit_transform")
                    .add(caller="fit_predict", callee="fit_transform")
                )
            else:
                (
                    method_mapping.add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="fit_transform", callee="fit")
                    .add(caller="fit_transform", callee="transform")
                    .add(caller="fit_predict", callee="fit")
                    .add(caller="fit_predict", callee="transform")
                )

            (
                method_mapping.add(caller="predict", callee="transform")
                .add(caller="predict", callee="transform")
                .add(caller="predict_proba", callee="transform")
                .add(caller="decision_function", callee="transform")
                .add(caller="predict_log_proba", callee="transform")
                .add(caller="transform", callee="transform")
                .add(caller="inverse_transform", callee="inverse_transform")
                .add(caller="score", callee="transform")
            )

            router.add(method_mapping=method_mapping, **{name: trans})

        # Then we add the last step
        if len(self.steps) > 0:
            final_name, final_est = self.steps[-1]
            if final_est is not None and final_est != "passthrough":
                # then we add the last step
                method_mapping = MethodMapping()
                if hasattr(final_est, "fit_transform"):
                    method_mapping.add(caller="fit_transform", callee="fit_transform")
                else:
                    method_mapping.add(caller="fit", callee="fit").add(
                        caller="fit", callee="transform"
                    )
                (
                    method_mapping.add(caller="fit", callee="fit")
                    .add(caller="predict", callee="predict")
                    .add(caller="fit_predict", callee="fit_predict")
                    .add(caller="predict_proba", callee="predict_proba")
                    .add(caller="decision_function", callee="decision_function")
                    .add(caller="predict_log_proba", callee="predict_log_proba")
                    .add(caller="transform", callee="transform")
                    .add(caller="inverse_transform", callee="inverse_transform")
                    .add(caller="score", callee="score")
                )

                router.add(method_mapping=method_mapping, **{final_name: final_est})

        return router

    def fit(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **params,
    ) -> Self:
        """Fit the pipeline.

        Fit all the transformers one after the other and sequentially
        transform the data. Finally, fit the transformed data using the
        final estimator.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            `X` is ignored.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`.

        **params
            Parameters requested and accepted by steps. Each step must
            have requested certain metadata for these parameters to be
            forwarded to them.

        Returns
        -------
        self
            Pipeline with fitted steps.

        """
        routed_params = self._check_method_params(method="fit", props=params)
        Xt, yt = self._fit(X, y, routed_params)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator is not None and self._final_estimator != "passthrough":
                with adjust(self._final_estimator, verbose=self._verbose):
                    self._mem_fit(
                        estimator=self._final_estimator,
                        X=Xt,
                        y=yt,
                        **routed_params[self.steps[-1][0]].fit,
                    )

        return self

    @available_if(_can_transform)
    def fit_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Fit the pipeline and transform the data.

        Call `fit` followed by `transform` on each transformer in the
        pipeline. The transformed data are finally passed to the final
        estimator that calls the `transform` method. Only valid if the
        final estimator implements `transform`. This also works when the
        final estimator is `None`, in which case all prior
        transformations are applied.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            `X` is ignored. None
            if the estimator only uses y.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`.

        **params
            Parameters requested and accepted by steps. Each step must
            have requested certain metadata for these parameters to be
            forwarded to them.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        routed_params = self._check_method_params(method="fit_transform", props=params)
        Xt, yt = self._fit(X, y, routed_params)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator is None or self._final_estimator == "passthrough":
                return variable_return(Xt, yt)

            with adjust(self._final_estimator, verbose=self._verbose):
                Xt, yt, _ = self._mem_fit_transform(
                    transformer=self._final_estimator,
                    X=Xt,
                    y=yt,
                    **routed_params[self.steps[-1][0]].fit_transform,
                )

        return variable_return(self._convert(Xt), self._convert(yt))

    @available_if(_can_transform)
    def transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        *,
        filter_train_only: Bool = True,
        **params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Transform the data.

        Call `transform` on each transformer in the pipeline. The
        transformed data are finally passed to the final estimator
        that calls the `transform` method. Only valid if the final
        estimator implements `transform`. This also works when the
        final estimator is `None`, in which case all prior
        transformations are applied.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            `X` is ignored. None if the pipeline only uses y.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`.

        filter_train_only: bool, default=True
            Whether to exclude transformers that should only be used
            on the training set.

        **params
            Parameters requested and accepted by steps. Each step must
            have requested certain metadata for these parameters to be
            forwarded to them.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        if X is None and y is None:
            raise ValueError("X and y cannot be both None.")

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        _raise_for_params(params, self, "transform")

        routed_params = process_routing(self, "transform", **params)
        for _, name, transformer in self._iter(filter_train_only=filter_train_only):
            with adjust(transformer, verbose=self._verbose):
                Xt, yt = self._mem_transform(
                    transformer=transformer,
                    X=Xt,
                    y=yt,
                    **routed_params[name].transform,
                )

        return variable_return(self._convert(Xt), self._convert(yt))

    @available_if(_can_inverse_transform)
    def inverse_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        *,
        filter_train_only: Bool = True,
        **params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Inverse transform for each step in a reverse order.

        All estimators in the pipeline must implement the
        `inverse_transform` method.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            `X` is ignored. None if the pipeline only uses y.

        y: sequence, dataframe-like or None, default=None
            Target column(s) corresponding to `X`.

        filter_train_only: bool, default=True
            Whether to exclude transformers that should only be used
            on the training set.

        **params
            Parameters requested and accepted by steps. Each step must
            have requested certain metadata for these parameters to be
            forwarded to them.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        if X is None and y is None:
            raise ValueError("X and y cannot be both None.")

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        _raise_for_params(params, self, "inverse_transform")

        routed_params = process_routing(self, "inverse_transform", **params)
        reverse_iter = reversed(list(self._iter(filter_train_only=filter_train_only)))
        for _, name, transformer in reverse_iter:
            with adjust(transformer, verbose=self._verbose):
                Xt, yt = self._mem_transform(
                    transformer=transformer,
                    X=Xt,
                    y=yt,
                    method="inverse_transform",
                    **routed_params[name].inverse_transform,
                )

        return variable_return(self._convert(Xt), self._convert(yt))

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X: XConstructor, **params) -> np.ndarray:
        """Transform, then decision_function of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        **params
            Parameters requested and accepted by steps. Each step must
            have requested certain metadata for these parameters to be
            forwarded to them.

        Returns
        -------
        np.ndarray
            Predicted confidence scores with shape=(n_samples,) for
            binary classification tasks (log likelihood ratio of the
            positive class) or shape=(n_samples, n_classes) for
            multiclass classification tasks.

        """
        Xt = to_df(X)

        _raise_for_params(params, self, "decision_function")

        routed_params = process_routing(self, "decision_function", **params)

        for _, name, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(
                    transformer=transformer,
                    X=Xt,
                    **routed_params.get(name, {}).get("transform", {}),
                )

        return self.steps[-1][1].decision_function(
            Xt, **routed_params.get(self.steps[-1][0], {}).get("decision_function", {})
        )

    @available_if(_final_estimator_has("predict"))
    def predict(
        self,
        X: XConstructor | None = None,
        fh: FHConstructor | None = None,
        **params,
    ) -> np.ndarray | Pandas:
        """Transform, then predict of the final estimator.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). Can only
            be `None` for [forecast][time-series] tasks.

        fh: int, sequence or [ForecastingHorizon][] or None, default=None
            The forecasting horizon encoding the time stamps to
            forecast at. Only for [forecast][time-series] tasks.

        **params
            Parameters requested and accepted by steps. Each step must
            have requested certain metadata for these parameters to be
            forwarded to them. Note that while this may be used to
            return uncertainties from some models with `return_std` or
            `return_cov`, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

        Returns
        -------
        np.ndarray, series or dataframe
            Predictions with shape=(n_samples,) or shape=(n_samples,
            n_targets) for [multioutput tasks][].

        """
        if X is None and fh is None:
            raise ValueError("X and fh cannot be both None.")

        Xt = to_df(X)

        routed_params = process_routing(self, "predict", **params)

        for _, name, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt, **routed_params[name].transform)

        if isinstance(self._final_estimator, BaseForecaster):
            if fh is None:
                raise ValueError("The fh parameter cannot be None for forecasting estimators.")

            return self.steps[-1][1].predict(fh=fh, X=Xt)
        else:
            return self.steps[-1][1].predict(Xt, **routed_params[self.steps[-1][0]].predict)

    @available_if(_final_estimator_has("predict_interval"))
    def predict_interval(
        self,
        fh: FHConstructor,
        X: XConstructor | None = None,
        *,
        coverage: Float | Sequence[Float] = 0.9,
    ) -> pd.DataFrame:
        """Transform, then predict_quantiles of the final estimator.

        Parameters
        ----------
        fh: int, sequence or [ForecastingHorizon][]
            The forecasting horizon encoding the time stamps to
            forecast at.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to `fh`.

        coverage: float or sequence, default=0.9
            Nominal coverage(s) of predictive interval(s).

        Returns
        -------
        dataframe
            Computed interval forecasts.

        """
        Xt = to_df(X)

        for _, _, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt)

        return self.steps[-1][1].predict_interval(fh=fh, X=Xt, coverage=coverage)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X: XConstructor, **params) -> np.ndarray:
        """Transform, then predict_log_proba of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        **params
            Parameters requested and accepted by steps. Each step must
            have requested certain metadata for these parameters to be
            forwarded to them.

        Returns
        -------
        list or np.ndarray
            Predicted class log-probabilities with shape=(n_samples,
            n_classes) or a list of arrays for [multioutput tasks][].

        """
        Xt = to_df(X)

        routed_params = process_routing(self, "predict_log_proba", **params)

        for _, name, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt, **routed_params[name].transform)

        return self.steps[-1][1].predict_log_proba(
            Xt, **routed_params[self.steps[-1][0]].predict_log_proba
        )

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(
        self,
        X: XConstructor | None = None,
        fh: FHConstructor | None = None,
        *,
        marginal: Bool = True,
        **params,
    ) -> list[np.ndarray] | np.ndarray | Normal:
        """Transform, then predict_proba of the final estimator.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). Can only
            be `None` for [forecast][time-series] tasks.

        fh: int, sequence, [ForecastingHorizon][] or None, default=None
            The forecasting horizon encoding the time stamps to
            forecast at. Only for [forecast][time-series] tasks.

        marginal: bool, default=True
            Whether returned distribution is marginal by time index.
            Only for [forecast][time-series] tasks.

        **params
            Parameters requested and accepted by steps. Each step must
            have requested certain metadata for these parameters to be
            forwarded to them.

        Returns
        -------
        list, np.ndarray or sktime.proba.[Normal][]

            - For classification tasks: Predicted class probabilities
              with shape=(n_samples, n_classes).
            - For [multioutput tasks][]: A list of arrays with
              shape=(n_samples, n_classes).
            - For [forecast][time-series] tasks: Distribution object.

        """
        if X is None and fh is None:
            raise ValueError("X and fh cannot be both None.")

        Xt = to_df(X)

        routed_params = process_routing(self, "predict_proba", **params)

        for _, name, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt, **routed_params[name].transform)

        if isinstance(self._final_estimator, BaseForecaster):
            if fh is None:
                raise ValueError("The fh parameter cannot be None for forecasting estimators.")

            return self.steps[-1][1].predict_proba(fh=fh, X=Xt, marginal=marginal)
        else:
            return self.steps[-1][1].predict_proba(
                Xt, **routed_params[self.steps[-1][0]].predict_proba
            )

    @available_if(_final_estimator_has("predict_quantiles"))
    def predict_quantiles(
        self,
        fh: FHConstructor,
        X: XConstructor | None = None,
        *,
        alpha: Float | Sequence[Float] = (0.05, 0.95),
    ) -> Pandas:
        """Transform, then predict_quantiles of the final estimator.

        Parameters
        ----------
        fh: int, sequence or [ForecastingHorizon][]
            The forecasting horizon encoding the time stamps to
            forecast at.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to `fh`.

        alpha: float or sequence, default=(0.05, 0.95)
            A probability or list of, at which quantile forecasts are
            computed.

        Returns
        -------
        dataframe
            Computed quantile forecasts.

        """
        Xt = to_df(X)

        for _, _, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt)

        return self.steps[-1][1].predict_quantiles(fh=fh, X=Xt, alpha=alpha)

    @available_if(_final_estimator_has("predict_residuals"))
    def predict_residuals(
        self,
        y: YConstructor,
        X: XConstructor | None = None,
    ) -> Pandas:
        """Transform, then predict_residuals of the final estimator.

        Parameters
        ----------
        y: sequence or dataframe
            Ground truth observations.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to `y`.

        Returns
        -------
        series or dataframe
            Residuals with shape=(n_samples,) or shape=(n_samples,
            n_targets) for [multivariate][] tasks.

        """
        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        for _, _, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, yt = self._mem_transform(transformer, Xt, yt)

        return self.steps[-1][1].predict_residuals(y=yt, X=Xt)

    @available_if(_final_estimator_has("predict_var"))
    def predict_var(
        self,
        fh: FHConstructor,
        X: XConstructor | None = None,
        *,
        cov: Bool = False,
    ) -> pd.DataFrame:
        """Transform, then predict_var of the final estimator.

        Parameters
        ----------
        fh: int, sequence or [ForecastingHorizon][]
            The forecasting horizon encoding the time stamps to
            forecast at.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to `fh`.

        cov: bool, default=False
            Whether to compute covariance matrix forecast or marginal
            variance forecasts.

        Returns
        -------
        dataframe
            Computed variance forecasts.

        """
        Xt = to_df(X)

        for _, _, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt)

        return self.steps[-1][1].predict_var(fh=fh, X=Xt, cov=cov)

    def set_output(self, *, transform: EngineDataOptions | None = None) -> Self:
        """Set output container.

        See sklearn's [user guide][set_output] on how to use the
        `set_output` API. See [here][data-acceleration] a description
        of the choices.

        Parameters
        ----------
        transform: str or None, default=None
            Configure the output of the `transform`, `fit_transform`,
            and `inverse_transform` method. If None, the configuration
            is not changed. Choose from:

            - "numpy"
            - "pandas" (default)
            - "pandas-pyarrow"
            - "polars"
            - "polars-lazy"
            - "pyarrow"
            - "modin"
            - "dask"
            - "pyspark"
            - "pyspark-pandas"

        Returns
        -------
        Self
            Estimator instance.

        """
        if transform is not None:
            self._engine = EngineTuple(data=transform)

        return self

    @available_if(_final_estimator_has("score"))
    def score(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        fh: FHConstructor | None = None,
        *,
        sample_weight: Sequence[Scalar] | None = None,
        **params,
    ) -> Float:
        """Transform, then score of the final estimator.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). Can only
            be `None` for [forecast][time-series] tasks.

        y: sequence, dataframe-like or None, default=None
            Target values corresponding to `X`.

        fh: int, sequence, [ForecastingHorizon][] or None, default=None
            The forecasting horizon encoding the time stamps to score.

        sample_weight: sequence or None, default=None
            Sample weights corresponding to `y` passed to the `score`
            method of the final estimator. If None, no sampling weight
            is performed. Only for non-forecast tasks.

        Returns
        -------
        float
            Mean accuracy, r2 or mape of self.predict(X) with respect
            to `y` (depending on the task).

        """
        if X is None and y is None:
            raise ValueError("X and y cannot be both None.")

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        # Drop sample weights if sktime estimator
        if not isinstance(self._final_estimator, BaseForecaster):
            params["sample_weight"] = sample_weight

        routed_params = process_routing(self, "score", **params)

        for _, name, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, yt = self._mem_transform(transformer, Xt, yt, **routed_params[name].transform)

        if isinstance(self._final_estimator, BaseForecaster):
            return self.steps[-1][1].score(y=yt, X=Xt, fh=fh)
        else:
            return self.steps[-1][1].score(Xt, yt, **routed_params[self.steps[-1][0]].score)

"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the ATOM's custom sklearn-like pipeline.

"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
from joblib import Memory
from sklearn.base import clone
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.pipeline import _final_estimator_has
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory
from sktime.proba.normal import Normal
from typing_extensions import Self

from atom.utils.types import (
    Bool, DataFrame, Estimator, FHConstructor, Float, Pandas, Scalar, Sequence,
    Verbose, XConstructor, YConstructor,
)
from atom.utils.utils import (
    NotFittedError, adjust_verbosity, check_is_fitted, fit_one,
    fit_transform_one, sign, transform_one, variable_return,
)


class Pipeline(SkPipeline):
    """Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be transformsers, that
    is, they must implement `fit` and `transform` methods. The final
    estimator only needs to implement `fit`. The transformers in the
    pipeline can be cached using the `memory` parameter.

    The purpose of the pipeline is to assemble several steps that can
    be cross-validated together while setting different parameters. For
    this, it enables setting parameters of the various steps using their
    names and the parameter name separated by `__`, as in the example
    below. A step's estimator may be replaced entirely by setting the
    parameter with its name to another estimator, or a transformer
    removed by setting it to `passthrough` or `None`.

    Read more in sklearn's the [user guide][pipelinedocs].

    !!! info
        This class behaves similarly to sklearn's [pipeline][skpipeline],
        and additionally:

        - Works with an empty pipeline.
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
        elapsed while fitting each step is printed.

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

    # Get the pipeline and make predictions
    pl = atom.lr.export_pipeline()
    print(pl.predict(X))
    ```

    """

    def __init__(
        self,
        steps: list[tuple[str, Estimator]],
        *,
        memory: str | Memory | None = None,
        verbose: Verbose | None = 0,
    ):
        super().__init__(steps, memory=memory, verbose=verbose)

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
        self._mem_fit = self._memory.cache(fit_transform_one)
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

        filter_passthrough: bool, default=True
            Whether to exclude estimators that should only be used for
            training (have the `_train_only` attribute).

        Returns
        -------
        int
            Index position in the pipeline.

        str
            Name of the estimator.

        Estimator
            Transformer or predictor instance.

        """
        it = super()._iter(with_final=with_final, filter_passthrough=filter_passthrough)
        if filter_train_only:
            return (x for x in it if not getattr(x[-1], "_train_only", False))
        else:
            return it

    def _fit(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params_steps,
    ) -> tuple[DataFrame | None, Pandas | None]:
        """Get data transformed through the pipeline.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. None if the pipeline only uses y.

        y: dict, sequence, dataframe or None, default=None
            Target column corresponding to `X`.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        dataframe or None
            Transformed feature set.

        series, dataframe or None
            Transformed target column.

        """
        self.steps: list[tuple[str, Estimator]] = list(self.steps)
        self._validate_steps()

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

            with adjust_verbosity(cloned, self.verbose):
                # Fit or load the current estimator from cache
                X, y, fitted_transformer = self._mem_fit(
                    transformer=cloned,
                    X=X,
                    y=y,
                    message=self._log_message(step),
                    **fit_params_steps.get(name, {}),
                )

            # Replace the estimator of the step with the fitted
            # estimator (necessary when loading from cache)
            self.steps[step] = (name, fitted_transformer)

        return X, y

    def fit(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> Self:
        """Fit the pipeline.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: dict, sequence, dataframe or None, default=None
            Target column corresponding to `X`.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        self
            Estimator instance.

        """
        fit_params_steps = self._check_fit_params(**fit_params)
        X, y = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            last_step = self._final_estimator
            if last_step is not None and last_step != "passthrough":
                with adjust_verbosity(last_step, self.verbose):
                    fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                    fit_one(last_step, X, y, **fit_params_last_step)

        return self

    @available_if(_can_transform)
    def fit_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> Pandas | tuple[DataFrame, Pandas]:
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
            X is ignored. None
            if the estimator only uses y.

        y: dict, sequence, dataframe or None, default=None
            Target column corresponding to `X`.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        fit_params_steps = self._check_fit_params(**fit_params)
        X, y = self._fit(X, y, **fit_params_steps)

        # Don't clone when caching is disabled to preserve backward compatibility
        if self.memory.location is None:
            last_step = self._final_estimator
        else:
            last_step = clone(self._final_estimator)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step is None or last_step == "passthrough":
                return variable_return(X, y)

            with adjust_verbosity(last_step, self.verbose):
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                X, y, _ = self._mem_fit(last_step, X, y, **fit_params_last_step)

        return variable_return(X, y)

    @available_if(_can_transform)
    def transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **kwargs,
    ) -> Pandas | tuple[DataFrame, Pandas]:
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
            X is ignored. None if the pipeline only uses y.

        y: dict, sequence, dataframe or None, default=None
            Target column corresponding to `X`.

        **kwargs
            Additional keyword arguments for the `_iter` inner method.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        if X is None and y is None:
            raise ValueError("X and y cannot be both None.")

        for _, _, transformer in self._iter(**kwargs):
            with adjust_verbosity(transformer, self.verbose):
                X, y = self._mem_transform(transformer, X, y)

        return variable_return(X, y)

    @available_if(_can_inverse_transform)
    def inverse_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Inverse transform for each step in a reverse order.

        All estimators in the pipeline must implement the
        `inverse_transform` method.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. None if the pipeline only uses y.

        y: dict, sequence, dataframe or None, default=None
            Target column corresponding to `X`.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        if X is None and y is None:
            raise ValueError("X and y cannot be both None.")

        for _, _, transformer in reversed(list(self._iter())):
            with adjust_verbosity(transformer, self.verbose):
                X, y = self._mem_transform(transformer, X, y, method="inverse_transform")

        return variable_return(X, y)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X: XConstructor) -> np.ndarray:
        """Transform, then decision_function of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted confidence scores with shape=(n_samples,) for
            binary classification tasks (log likelihood ratio of the
            positive class) or shape=(n_samples, n_classes) for
            multiclass classification tasks.

        """
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, _ = self._mem_transform(transformer, X)

        return self.steps[-1][1].decision_function(X)

    @available_if(_final_estimator_has("predict"))
    def predict(
        self,
        X: XConstructor | None = None,
        fh: FHConstructor | None = None,
        **predict_params,
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

        **predict_params
            Additional keyword arguments for the predict method. Note
            that while this may be used to return uncertainties from
            some models with return_std or return_cov, uncertainties
            that are generated by the transformations in the pipeline
            are not propagated to the final estimator.

        Returns
        -------
        np.ndarray, series or dataframe
            Predictions with shape=(n_samples,) or shape=(n_samples,
            n_targets) for [multioutput tasks][].

        """
        if X is None and fh is None:
            raise ValueError("X and fh cannot be both None.")

        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, _ = self._mem_transform(transformer, X)

        if "fh" in sign(self.steps[-1][1].predict):
            if fh is None:
                raise ValueError("The fh parameter cannot be None for forecasting estimators.")

            return self.steps[-1][1].predict(fh=fh, X=X, **predict_params)
        else:
            return self.steps[-1][1].predict(X, **predict_params)

    @available_if(_final_estimator_has("predict_interval"))
    def predict_interval(
        self,
        fh: FHConstructor,
        X: XConstructor | None = None,
        *,
        coverage: Float | Sequence[Float] = 0.9,
    ) -> Pandas:
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
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, y = self._mem_transform(transformer, X)

        return self.steps[-1][1].predict_interval(fh=fh, X=X, coverage=coverage)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X: XConstructor) -> np.ndarray:
        """Transform, then predict_log_proba of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        list or np.ndarray
            Predicted class log-probabilities with shape=(n_samples,
            n_classes) or a list of arrays for [multioutput tasks][].

        """
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, _ = self._mem_transform(transformer, X)

        return self.steps[-1][1].predict_log_proba(X)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(
        self,
        X: XConstructor | None = None,
        fh: FHConstructor | None = None,
        *,
        marginal: Bool = True,
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

        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, _ = self._mem_transform(transformer, X)

        if "fh" in sign(self.steps[-1][1].predict_proba):
            if fh is None:
                raise ValueError("The fh parameter cannot be None for forecasting estimators.")

            return self.steps[-1][1].predict_proba(fh=fh, X=X, marginal=marginal)
        else:
            return self.steps[-1][1].predict_proba(X)

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
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, y = self._mem_transform(transformer, X)

        return self.steps[-1][1].predict_quantiles(fh=fh, X=X, alpha=alpha)

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
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, y = self._mem_transform(transformer, X, y)

        return self.steps[-1][1].predict_residuals(y=y, X=X)

    @available_if(_final_estimator_has("predict_var"))
    def predict_var(
        self,
        fh: FHConstructor,
        X: XConstructor | None = None,
        *,
        cov: Bool = False,
    ) -> DataFrame:
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
        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, _ = self._mem_transform(transformer, X)

        return self.steps[-1][1].predict_var(fh=fh, X=X, cov=cov)

    @available_if(_final_estimator_has("score"))
    def score(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        fh: FHConstructor | None = None,
        *,
        sample_weight: Sequence[Scalar] | None = None,
    ) -> Float:
        """Transform, then score of the final estimator.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). Can only
            be `None` for [forecast][time-series] tasks.

        y: dict, sequence, dataframe or None, default=None
            Target values corresponding to `X`.

        fh: int, sequence, [ForecastingHorizon][] or None, default=None
            The forecasting horizon encoding the time stamps to score.

        sample_weight: sequence or None, default=None
            Sample weights corresponding to y.

        Returns
        -------
        float
            Mean accuracy, r2 or mape of self.predict(X) with respect
            to y.

        """
        if X is None and y is None:
            raise ValueError("X and y cannot be both None.")

        for _, _, transformer in self._iter(with_final=False):
            with adjust_verbosity(transformer, self.verbose):
                X, y = self._mem_transform(transformer, X, y)

        if "fh" in sign(self.steps[-1][1].score):
            return self.steps[-1][1].score(y=y, X=X, fh=fh)
        else:
            return self.steps[-1][1].score(X, y, sample_weight=sample_weight)

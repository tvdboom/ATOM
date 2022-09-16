# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the ATOM's custom sklearn-like pipeline.

"""

from sklearn import pipeline
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory

from atom.utils import (
    check_is_fitted, fit_one, fit_transform_one, transform_one,
    variable_return,
)


def _final_estimator_has(attr):
    """Check that final_estimator has attribute `attr`.

    Used together with `available_if` in Pipeline.

    """

    def check(self):
        # Raise original `AttributeError` if `attr` does not exist
        getattr(self._final_estimator, attr)
        return True

    return check


class Pipeline(pipeline.Pipeline):
    """Custom Pipeline class.

    This class behaves as a sklearn pipeline, and additionally:

    - Always outputs pandas objects.
    - Is able to transform only X or y.
    - Accepts transformers that change the target column.
    - Accepts transformers that drop rows.
    - Accepts transformers that only are fitted on a subset
      of the provided dataset.
    - Uses transformers that are only applied on the training set
      to fit the pipeline, not to make predictions on new data.
    - The instance is considered fitted at initialization if all
      the underlying transformers/estimator in the pipeline are.

    Note: This Pipeline only works with estimators whose parameters
    for fit, transform, predict, etc... are named X and/or y.

    """

    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)

        # If all estimators are fitted, Pipeline is fitted
        self._is_fitted = False
        if all(check_is_fitted(est[2], False) for est in self._iter(True, True, False)):
            self._is_fitted = True

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        """Set up cache memory objects."""
        self._memory = check_memory(value)
        self._memory_fit = self._memory.cache(fit_transform_one)
        self._memory_transform = self._memory.cache(transform_one)

    def _can_transform(self):
        return (
            self._final_estimator == "passthrough"
            or hasattr(self._final_estimator, "transform")
        )

    def _can_inverse_transform(self):
        return all(hasattr(t, "inverse_transform") for _, _, t in self._iter())

    def _iter(self, with_final=True, filter_passthrough=True, filter_train_only=True):
        """Generate (idx, (name, trans)) tuples from self.steps.

        By default, estimators that are only applied on the training
        set are filtered out for predictions.

        """
        it = super()._iter(with_final, filter_passthrough)
        if filter_train_only:
            return filter(lambda x: not getattr(x[-1], "_train_only", False), it)
        else:
            return it

    def _fit(self, X=None, y=None, **fit_params_steps):
        self.steps = list(self.steps)
        self._validate_steps()

        for (step_idx, name, transformer) in self._iter(False, False, False):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(transformer, "transform"):
                if self._memory_fit.__class__.__name__ == "NotMemorizedFunc":
                    # Don't clone when caching is disabled to
                    # preserve backward compatibility
                    cloned = transformer
                else:
                    cloned = clone(transformer)

                    # Attach internal attrs otherwise wiped by clone
                    for attr in ("_cols", "_train_only"):
                        if hasattr(transformer, attr):
                            setattr(cloned, attr, getattr(transformer, attr))

                # Fit or load the current estimator from cache
                X, y, fitted_transformer = self._memory_fit(
                    transformer=cloned,
                    X=X,
                    y=y,
                    message=self._log_message(step_idx),
                    **fit_params_steps[name],
                )

            # Replace the estimator of the step with the fitted
            # estimator (necessary when loading from the cache)
            self.steps[step_idx] = (name, fitted_transformer)

        return X, y

    def fit(self, X=None, y=None, **fit_params):
        """Fit the pipeline.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. None
            if the pipeline only uses y.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        Pipeline
            Estimator instance.

        """
        fit_params_steps = self._check_fit_params(**fit_params)
        X, y = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                fit_one(self._final_estimator, X, y, **fit_params_last_step)

        self._is_fitted = True
        return self

    @available_if(_can_transform)
    def transform(self, X=None, y=None):
        """Transform the data.

        Call `transform` of each transformer in the pipeline. The
        transformed data are finally passed to the final estimator
        that calls the `transform` method. Only valid if the final
        estimator implements `transform`. This also works where final
        estimator is `None` in which case all prior transformations
        are applied.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. None
            if the pipeline only uses y.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        pd.DataFrame
            Transformed feature set. Only returned if provided.

        pd.Series
            Transformed target column. Only returned if provided.

        """
        for _, _, transformer in self._iter():
            X, y = self._memory_transform(transformer, X, y)

        return variable_return(X, y)

    def fit_transform(self, X=None, y=None, **fit_params):
        """Fit the pipeline and transform the data.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. None
            if the estimator only uses y.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        np.array
            Transformed dataset.

        """
        fit_params_steps = self._check_fit_params(**fit_params)
        X, y = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return variable_return(X, y)

            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            X, y, _ = fit_transform_one(last_step, X, y, **fit_params_last_step)

        return variable_return(X, y)

    @available_if(_can_inverse_transform)
    def inverse_transform(self, X=None, y=None):
        """Inverse transform for each step in a reverse order.

        All estimators in the pipeline must implement the
        `inverse_transform` method.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. None
            if the pipeline only uses y.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        pd.DataFrame
            Transformed feature set. Only returned if provided.

        pd.Series
            Transformed target column. Only returned if provided.

        """
        for _, _, transformer in reversed(list(self._iter())):
            X, y = self._memory_transform(transformer, X, y, method="inverse_transform")

        return variable_return(X, y)

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Transform, then predict of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        **predict_params
            Additional keyword arguments for the predict method. Note
            that while this may be used to return uncertainties from
            some models with return_std or return_cov, uncertainties
            that are generated by the transformations in the pipeline
            are not propagated to the final estimator.

        Returns
        -------
        np.array
            Predicted target with shape=(n_samples,).

        """
        for _, name, transformer in self._iter(with_final=False):
            X, _ = self._memory_transform(transformer, X)

        return self.steps[-1][-1].predict(X, **predict_params)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Transform, then predict_proba of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.array
            Predicted class probabilities.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, _ = self._memory_transform(transformer, X)

        return self.steps[-1][-1].predict_proba(X)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Transform, then predict_log_proba of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.array
            Predicted class log-probabilities.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, _ = self._memory_transform(transformer, X)

        return self.steps[-1][-1].predict_log_proba(X)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X):
        """Transform, then decision_function of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.array
            Predicted confidence scores.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, _ = self._memory_transform(transformer, X)

        return self.steps[-1][-1].decision_function(X)

    @available_if(_final_estimator_has("score"))
    def score(self, X, y, sample_weight=None):
        """Transform, then score of the final estimator.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        sample_weight: sequence or None, default=None
            Sample weights corresponding to y.

        Returns
        -------
        float
            Mean accuracy or r2 of self.predict(X) with respect to y.

        """
        for _, _, transformer in self._iter(with_final=False):
            X, y = self._memory_transform(transformer, X, y)

        return self.steps[-1][-1].score(X, y, sample_weight=sample_weight)

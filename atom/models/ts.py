"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing all time series models.

"""

from __future__ import annotations

from typing import Any, ClassVar

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution as Cat
from optuna.distributions import FloatDistribution as Float
from optuna.distributions import IntDistribution as Int
from optuna.trial import Trial

from atom.basemodel import ForecastModel
from atom.utils.types import Predictor


class ARIMA(ForecastModel):
    """Autoregressive Integrated Moving Average Model.

    Seasonal ARIMA models and exogenous input is supported, hence this
    estimator is capable of fitting SARIMA, ARIMAX, and SARIMAX.

    An ARIMA model is a generalization of an autoregressive moving
    average (ARMA) model, and is fitted to time-series data in an effort
    to forecast future points. ARIMA models can be especially
    efficacious in cases where data shows evidence of non-stationarity.

    The "AR" part of ARIMA indicates that the evolving variable of
    interest is regressed on its own lagged (i.e., prior observed)
    values. The "MA" part indicates that the regression error is
    actually a linear combination of error terms whose values occurred
    contemporaneously and at various times in the past. The "I" (for
    "integrated") indicates that the data values have been replaced with
    the difference between their values and the previous values (and this
    differencing process may have been performed more than once).

    Corresponding estimators are:

    - [ARIMA][arimaclass] for forecasting tasks.

    !!! warning
        ARIMA often runs into numerical errors when optimizing the
        hyperparameters. Possible solutions are:

        - Use the [AutoARIMA][] model instead.
        - Use [`est_params`][directforecaster-est_params] to specify the
          orders manually, e.g., `#!python atom.run("arima", n_trials=5,
          est_params={"order": (1, 1, 0)})`.
        - Use the `catch` parameter in [`ht_params`][directforecaster-ht_params]
          to avoid raising every exception, e.g., `#!python atom.run("arima",
          n_trials=5, ht_params={"catch": (Exception,)})`.

    See Also
    --------
    atom.models:AutoARIMA

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    atom = ATOMForecaster(X)
    atom.run(models="ARIMA", verbose=2)
    ```

    """

    acronym = "ARIMA"
    handles_missing = True
    uses_exogenous = True
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.arima"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "ARIMA"}

    _order = ("p", "d", "q")
    _s_order = ("P", "D", "Q")

    def _get_parameters(self, trial: Trial) -> dict[str, BaseDistribution]:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        dict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        # If no seasonal periodicity, set seasonal components to zero
        if not self._config.sp:
            for p in self._s_order:
                if p in params:
                    params[p] = 0

        return params

    def _trial_to_est(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert trial's hyperparameters to parameters for the estimator.

        Parameters
        ----------
        params: dict
            Trial's hyperparameters.

        Returns
        -------
        dict
            Estimator's hyperparameters.

        """
        params = super()._trial_to_est(params)

        # Convert params to hyperparameters 'order' and 'seasonal_order'
        if all(p in params for p in self._order) and "order" not in params:
            params["order"] = [params.pop(p) for p in self._order]
        else:
            for p in self._order:
                params.pop(p, None)

        if (
            all(p in params for p in self._s_order)
            and self._config.sp
            and "seasonal_order" not in params
        ):
            params["seasonal_order"] = [params.pop(p) for p in self._s_order] + [self._config.sp]
        else:
            for p in self._s_order:
                params.pop(p, None)

        return params

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = {
            "p": Int(0, 2),
            "d": Int(0, 1),
            "q": Int(0, 2),
            "P": Int(0, 2),
            "D": Int(0, 1),
            "Q": Int(0, 2),
            "method": Cat(
                ["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]
            ),
            "maxiter": Int(50, 200, step=10),
            "with_intercept": Cat([True, False]),
        }

        # Drop order and seasonal_order params if specified by user
        if "order" in self._est_params:
            for p in self._order:
                dist.pop(p)
        if "seasonal_order" in self._est_params:
            for p in self._s_order:
                dist.pop(p)

        return dist


class AutoARIMA(ForecastModel):
    """Automatic Autoregressive Integrated Moving Average Model.

    [ARIMA][] implementation that includes automated fitting of
    (S)ARIMA(X) hyperparameters (p, d, q, P, D, Q). The AutoARIMA
    algorithm seeks to identify the most optimal parameters for an
    ARIMA model, settling on a single fitted ARIMA model. This process
    is based on the commonly-used R function.

    AutoARIMA works by conducting differencing tests (i.e.,
    Kwiatkowski-Phillips-Schmidt-Shin, Augmented Dickey-Fuller or
    Phillips-Perron) to determine the order of differencing, d, and
    then fitting models within defined ranges. AutoARIMA also seeks
    to identify the optimal P and Q hyperparameters after conducting
    the Canova-Hansen to determine the optimal order of seasonal
    differencing.

    Note that due to stationarity issues, AutoARIMA might not find a
    suitable model that will converge. If this is the case, a ValueError
    is thrown suggesting stationarity-inducing measures be taken prior
    to re-fitting or that a new range of order values be selected.

    Corresponding estimators are:

    - [AutoARIMA][autoarimaclass] for forecasting tasks.

    See Also
    --------
    atom.models:ARIMA
    atom.models:ETS

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    atom = ATOMForecaster(X, random_state=1)
    atom.run(models="autoarima", verbose=2)
    ```

    """

    acronym = "AutoARIMA"
    handles_missing = True
    uses_exogenous = True
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.arima"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "AutoARIMA"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        methods = ["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]

        return {
            "method": Cat(methods),
            "maxiter": Int(50, 200, step=10),
            "with_intercept": Cat([True, False]),
        }


class BATS(ForecastModel):
    """BATS forecaster with multiple seasonality.

    BATS is acronym for:

    - Box-Cox transformation
    - ARMA errors
    - Trend
    - Seasonal components

    BATS was designed to forecast time series with multiple seasonal
    periods. For example, daily data may have a weekly pattern as well
    as an annual pattern. Or hourly data can have three seasonal periods:
    a daily pattern, a weekly pattern, and an annual pattern.

    In BATS, a [Box-Cox transformation][boxcox] is applied to the
    original time series, and then this is modeled as a linear
    combination of an exponentially smoothed trend, a seasonal
    component and an ARMA component. BATS conducts some hyperparameter
    tuning (e.g., which of these components to keep and which to discard)
    using AIC.

    Corresponding estimators are:

    - [BATS][batsclass] for forecasting tasks.

    See Also
    --------
    atom.models:ARIMA
    atom.models:AutoARIMA
    atom.models:TBATS

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="BATS", verbose=2)
    ```

    """

    acronym = "BATS"
    handles_missing = False
    uses_exogenous = False
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.bats"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "BATS"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Parameters
        ----------
        params: dict
            Hyperparameters for the estimator.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        return super()._get_est({"show_warnings": self.warnings != "ignore"} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "use_box_cox": Cat([True, False, None]),
            "use_trend": Cat([True, False, None]),
            "use_damped_trend": Cat([True, False, None]),
            "use_arma_errors": Cat([True, False]),
        }


class Croston(ForecastModel):
    """Croston's method for forecasting.

    Croston's method is a modification of (vanilla) exponential
    smoothing to handle intermittent time series. A time series is
    considered intermittent if many of its values are zero and the
    gaps between non-zero entries are not periodic.

    Croston's method will predict a constant value for all future
    times, so Croston's method essentially provides another notion
    for the average value of a time series.

    Corresponding estimators are:

    - [Croston][crostonclass] for forecasting tasks.

    See Also
    --------
    atom.models:ExponentialSmoothing
    atom.models:ETS
    atom.models:NaiveForecaster

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="Croston", verbose=2)
    ```

    """

    acronym = "Croston"
    handles_missing = False
    uses_exogenous = True
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.croston"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "Croston"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {"smoothing": Float(0, 1, step=0.1)}


class ExponentialSmoothing(ForecastModel):
    """Holt-Winters Exponential Smoothing forecaster.

    Corresponding estimators are:

    - [ExponentialSmoothing][esclass] for forecasting tasks.

    See Also
    --------
    atom.models:ARIMA
    atom.models:ETS
    atom.models:PolynomialTrend

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="ES", verbose=2)
    ```

    """

    acronym = "ES"
    handles_missing = False
    uses_exogenous = False
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.exp_smoothing"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "ExponentialSmoothing"}

    def _get_parameters(self, trial: Trial) -> dict:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        dict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if not self._get_param("trend", params) and "damped_trend" in params:
            params["damped_trend"] = False

        return params

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "trend": Cat(["add", "mul", None]),
            "damped_trend": Cat([True, False]),
            "seasonal": Cat(["add", "mul", None]),
            "use_boxcox": Cat([True, False]),
            "initialization_method": Cat(["estimated", "heuristic"]),
            "method": Cat(["L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr", "bh", "ls"]),
            "use_brute": Cat([True, False]),
        }


class ETS(ForecastModel):
    """ETS model with automatic fitting capabilities.

    The ETS models are a family of time series models with an
    underlying state space model consisting of a level component,
    a trend component (T), a seasonal component (S), and an error
    term (E).

    Corresponding estimators are:

    - [AutoETS][] for forecasting tasks.

    See Also
    --------
    atom.models:ARIMA
    atom.models:ExponentialSmoothing
    atom.models:PolynomialTrend

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="ETS", verbose=2)

    ```

    """

    acronym = "ETS"
    handles_missing = True
    uses_exogenous = False
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.ets"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "AutoETS"}

    def _get_parameters(self, trial: Trial) -> dict:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        dict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if not self._get_param("trend", params) and "damped_trend" in params:
            params["damped_trend"] = False

        return params

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "error": Cat(["add", "mul"]),
            "trend": Cat(["add", "mul", None]),
            "damped_trend": Cat([True, False]),
            "seasonal": Cat(["add", "mul", None]),
            "initialization_method": Cat(["estimated", "heuristic"]),
            "maxiter": Int(500, 2000, step=100),
            "auto": Cat([True, False]),
            "information_criterion": Cat(["aic", "bic", "aicc"]),
            "allow_multiplicative_trend": Cat([True, False]),
            "restrict": Cat([True, False]),
            "additive_only": Cat([True, False]),
            "ignore_inf_ic": Cat([True, False]),
        }


class NaiveForecaster(ForecastModel):
    """Naive Forecaster.

    NaiveForecaster is a dummy forecaster that makes forecasts using
    simple strategies based on naive assumptions about past trends
    continuing. When used in [multivariate][] tasks, each column is
    forecasted with the same strategy.

    Corresponding estimators are:

    - [NaiveForecaster][naiveforecasterclass] for forecasting tasks.

    See Also
    --------
    atom.models:ExponentialSmoothing
    atom.models:Dummy
    atom.models:PolynomialTrend

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="NF", verbose=2)

    ```

    """

    acronym = "NF"
    handles_missing = True
    uses_exogenous = False
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.naive"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "NaiveForecaster"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {"strategy": Cat(["last", "mean", "drift"])}


class PolynomialTrend(ForecastModel):
    """Polynomial Trend forecaster.

    Forecast time series data with a polynomial trend, using a sklearn
    [LinearRegression][] class to regress values of time series on
    index, after extraction of polynomial features.

    Corresponding estimators are:

    - [PolynomialTrendForecaster][] for forecasting tasks.

    See Also
    --------
    atom.models:ARIMA
    atom.models:ETS
    atom.models:NaiveForecaster

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="PT", verbose=2)
    ```

    """

    acronym = "PT"
    handles_missing = False
    uses_exogenous = False
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.trend"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "PolynomialTrendForecaster"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "degree": Int(1, 5),
            "with_intercept": Cat([True, False]),
        }


class STL(ForecastModel):
    """Seasonal-Trend decomposition using Loess.

    STL is a technique commonly used for decomposing time series data
    into components like trend, seasonality, and residuals.

    Corresponding estimators are:

    - [STLForecaster][] for forecasting tasks.

    See Also
    --------
    atom.models:Croston
    atom.models:ETS
    atom.models:Theta

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="STL", verbose=2)
    ```

    """

    acronym = "STL"
    handles_missing = False
    uses_exogenous = False
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.trend"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "STLForecaster"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "seasonal": Int(3, 11, step=2),
            "seasonal_deg": Cat([0, 1]),
            "trend_deg": Cat([0, 1]),
            "low_pass_deg": Cat([0, 1]),
            "robust": Cat([True, False]),
        }


class TBATS(ForecastModel):
    """TBATS forecaster with multiple seasonality.

    TBATS is acronym for:

    - Trigonometric seasonality
    - Box-Cox transformation
    - ARMA errors
    - Trend
    - Seasonal components

    TBATS was designed to forecast time series with multiple seasonal
    periods. For example, daily data may have a weekly pattern as well
    as an annual pattern. Or hourly data can have three seasonal periods:
    a daily pattern, a weekly pattern, and an annual pattern.

    In BATS, a [Box-Cox transformation][boxcox] is applied to the
    original time series, and then this is modeled as a linear
    combination of an exponentially smoothed trend, a seasonal
    component and an ARMA component. The seasonal components are
    modeled by trigonometric functions via Fourier series. TBATS
    conducts some hyper-parameter tuning (e.g. which of these
    components to keep and which to discard) using AIC.

    Corresponding estimators are:

    - [TBATS][tbatsclass] for forecasting tasks.

    See Also
    --------
    atom.models:BATS
    atom.models:ARIMA
    atom.models:AutoARIMA

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="TBATS", verbose=2)
    ```

    """

    acronym = "TBATS"
    handles_missing = False
    uses_exogenous = False
    in_sample_prediction = True
    multiple_seasonality = True
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.tbats"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "TBATS"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Parameters
        ----------
        params: dict
            Hyperparameters for the estimator.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        return super()._get_est({"show_warnings": self.warnings != "ignore"} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "use_box_cox": Cat([True, False, None]),
            "use_trend": Cat([True, False, None]),
            "use_damped_trend": Cat([True, False, None]),
            "use_arma_errors": Cat([True, False]),
        }


class Theta(ForecastModel):
    """Theta method for forecasting.

    The theta method is equivalent to simple [ExponentialSmoothing][]
    with drift. The series is tested for seasonality, and, if deemed
    seasonal, the series is seasonally adjusted using a classical
    multiplicative decomposition before applying the theta method. The
    resulting forecasts are then reseasonalised.

    In cases where ES results in a constant forecast, the theta
    forecaster will revert to predicting the SES constant plus a linear
    trend derived from the training data.

    Prediction intervals are computed using the underlying state space
    model.

    Corresponding estimators are:

    - [ThetaForecaster][] for forecasting tasks.

    See Also
    --------
    atom.models:Croston
    atom.models:ExponentialSmoothing
    atom.models:PolynomialTrend

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="Theta", verbose=2)
    ```

    """

    acronym = "Theta"
    handles_missing = False
    uses_exogenous = False
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.theta"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "ThetaForecaster"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {"deseasonalize": Cat([False, True])}


class VAR(ForecastModel):
    """Vector Autoregressive.

    A VAR model is a generalization of the univariate autoregressive.

    Corresponding estimators are:

    - [VAR][varclass] for forecasting tasks.

    See Also
    --------
    atom.models:MSTL
    atom.models:Prophet
    atom.models:VARMAX

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="VAR", verbose=2)
    ```

    """

    acronym = "VAR"
    handles_missing = False
    uses_exogenous = False
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = True
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.var"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "VAR"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "trend": Cat(["c", "ct", "ctt", "n"]),
            "ic": Cat(["aic", "fpe", "hqic", "bic"]),
        }


class VARMAX(ForecastModel):
    """Vector Autoregressive Moving Average.

    Variation on the [VAR][] that makes use of the exogenous variables.

    Corresponding estimators are:

    - [VARMAX][varmaxclass] for forecasting tasks.

    See Also
    --------
    atom.models:MSTL
    atom.models:Prophet
    atom.models:VAR

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="VARMAX", verbose=2)
    ```

    """

    acronym = "VARMAX"
    handles_missing = False
    uses_exogenous = True
    in_sample_prediction = True
    multiple_seasonality = False
    native_multivariate = True
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.var"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "VARMAX"}

    _order = ("p", "q")

    def _trial_to_est(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert trial's hyperparameters to parameters for the estimator.

        Parameters
        ----------
        params: dict
            Trial's hyperparameters.

        Returns
        -------
        dict
            Estimator's hyperparameters.

        """
        params = super()._trial_to_est(params)

        # Convert params to hyperparameter 'order'
        if all(p in params for p in self._order) and "order" not in params:
            params["order"] = [params.pop(p) for p in self._order]
        else:
            for p in self._order:
                params.pop(p, None)

        return params

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Parameters
        ----------
        params: dict
            Hyperparameters for the estimator.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        return super()._get_est({"suppress_warnings": self.warnings == "ignore"} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "p": Int(0, 2),
            "q": Int(0, 2),
            "trend": Cat(["c", "ct", "ctt", "n"]),
            "error_cov_type": Cat(["diagonal", "unstructured"]),
            "measurement_error": Cat([True, False]),
            "enforce_stationarity": Cat([True, False]),
            "enforce_invertibility": Cat([True, False]),
            "cov_type": Cat(["opg", "oim", "approx", "robust", "robust_approx"]),
            "method": Cat(
                ["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]
            ),
            "maxiter": Int(50, 200, step=10),
            "optim_score": Cat(["harvey", "approx", None]),
            "optim_complex_step": Cat([True, False]),
            "optim_hessian": Cat(["opg", "oim", "approx"]),
        }

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
    atom.models:SARIMAX
    atom.models:VARMAX

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
        if all(p in params for p in self._order):
            params["order"] = [params[p] for p in self._order]
        if all(p in params for p in self._s_order) and self._config.sp:
            params["seasonal_order"] = [params[p] for p in self._s_order] + [self._config.sp]

        # Drop order and seasonal_order params
        for p in self._order:
            params.pop(p, None)
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
    atom.models:SARIMAX

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
        return {
            "method": Cat(
                ["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]
            ),
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


class DynamicFactor(ForecastModel):
    """Dynamic Factor.

    The DynamicFactor model incorporates dynamic factors to predict
    future values. In this context, "dynamic factors" refer to
    variables that change over time and impact the variable you are
    trying to forecast.

    !!! warning
        DynamicFactor only supports [multivariate][] tasks.

    Corresponding estimators are:

    - [DynamicFactor][dynamicfactorclass] for forecasting tasks.

    See Also
    --------
    atom.models:ExponentialSmoothing
    atom.models:LTS
    atom.models:PolynomialTrend

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="DF", verbose=2)

    ```

    """

    acronym = "DF"
    handles_missing = True
    uses_exogenous = True
    multiple_seasonality = False
    native_multivariate = True
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.dynamic_factor"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "DynamicFactor"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "k_factors": Int(1, 10),
            "error_cov_type": Cat(["scalar", "diagonal", "unstructured"]),
            "error_var": Cat([True, False]),
            "enforce_stationarity": Cat([True, False]),
            "cov_type": Cat(["opg", "oim", "approx", "robust", "robust_approx", "none"]),
            "method": Cat(
                ["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]
            ),
            "maxiter": Int(50, 200, step=10),
        }


class ExponentialSmoothing(ForecastModel):
    """Holt-Winters Exponential Smoothing forecaster.

    ExponentialSmoothing is a forecasting model that extends simple
    exponential smoothing to handle seasonality and trends in the
    data. This method is particularly useful for forecasting time
    series data with a systematic pattern that repeats over time.

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

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = {
            "error": Cat(["add", "mul"]),
            "trend": Cat(["add", "mul", None]),
            "damped_trend": Cat([True, False]),
            "seasonal": Cat(["add", "mul"]),
            "initialization_method": Cat(["estimated", "heuristic"]),
            "maxiter": Int(500, 2000, step=100),
            "auto": Cat([True, False]),
            "information_criterion": Cat(["aic", "bic", "aicc"]),
            "allow_multiplicative_trend": Cat([True, False]),
            "restrict": Cat([True, False]),
            "additive_only": Cat([True, False]),
            "ignore_inf_ic": Cat([True, False]),
        }

        if not self._config.sp:
            dist.pop("seasonal")

        return dist


class MSTL(ForecastModel):
    """Multiple Seasonal-Trend decomposition using LOESS model.

    The MSTL decomposes the time series in multiple seasonalities using
    LOESS. Then forecasts the trend using a custom non-seasonal model
    (trend_forecaster) and each seasonality using a SeasonalNaive model.

    Corresponding estimators are:

    - [StatsForecastMSTL][] for forecasting tasks.

    See Also
    --------
    atom.models:DynamicFactor
    atom.models:Prophet
    atom.models:VARMAX

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="MSTL", verbose=2)

    ```

    """

    acronym = "MSTL"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = True
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.statsforecast"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "StatsForecastMSTL"}

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
        return super()._get_est({"season_length": self._config.sp or 1} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "seasonal_deg": Cat([0, 1]),
            "trend_deg": Cat([0, 1]),
            "low_pass_deg": Cat([0, 1]),
            "robust": Cat([True, False]),
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


class Prophet(ForecastModel):
    """Prophet forecaster by Facebook.

    Prophet is designed to handle time series data with strong seasonal
    patterns, holidays, and missing data. Prophet is particularly useful
    for business applications where time series data may exhibit
    irregularities and is not always perfectly regular.

    Corresponding estimators are:

    - [Prophet][prophetclass] for forecasting tasks.

    See Also
    --------
    atom.models:DynamicFactor
    atom.models:MSTL
    atom.models:VARMAX

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="Prophet", verbose=2)

    ```

    """

    acronym = "Prophet"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = True
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.fbprophet"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "StatsForecastMSTL"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "seasonality_mode": Cat(["additive", "multiplicative"]),
            "changepoint_prior_scale": Float(0.001, 0.5, log=True),
            "seasonality_prior_scale": Float(0.001, 10, log=True),
            "holidays_prior_scale": Float(0.001, 10, log=True),
        }


class SARIMAX(ForecastModel):
    """Seasonal Autoregressive Integrated Moving Average with eXogenous factors.

    SARIMAX stands for Seasonal Autoregressive Integrated Moving Average
    with eXogenous factors. It extends [ARIMA][] by incorporating seasonal
    components and exogenous variables. Note that the ARIMA model is also
    capable of fitting SARIMAX.

    Corresponding estimators are:

    - [SARIMAX][sarimaxclass] for forecasting tasks.

    !!! warning
        SARIMAX often runs into numerical errors when optimizing the
        hyperparameters. Possible solutions are:

        - Use the [AutoARIMA][] model instead.
        - Use [`est_params`][directforecaster-est_params] to specify the
          orders manually, e.g., `#!python atom.run("sarimax", n_trials=5,
          est_params={"order": (1, 1, 0)})`.
        - Use the `catch` parameter in [`ht_params`][directforecaster-ht_params]
          to avoid raising every exception, e.g., `#!python atom.run("sarimax",
          n_trials=5, ht_params={"catch": (Exception,)})`.

    See Also
    --------
    atom.models:ARIMA
    atom.models:AutoARIMA
    atom.models:VARMAX

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    atom = ATOMForecaster(X)
    atom.run(models="SARIMAX", verbose=2)
    ```

    """

    acronym = "SARIMAX"
    handles_missing = False
    uses_exogenous = True
    multiple_seasonality = False
    native_multivariate = False
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.sarimax"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "SARIMAX"}

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
        if all(p in params for p in self._order):
            params["order"] = [params[p] for p in self._order]
        if all(p in params for p in self._s_order) and self._config.sp:
            params["seasonal_order"] = [params[p] for p in self._s_order] + [self._config.sp]

        # Drop order and seasonal_order params
        for p in self._order:
            params.pop(p, None)
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
            "trend": Cat(["n", "c", "t", "ct"]),
            "measurement_error": Cat([True, False]),
            "time_varying_regression": Cat([True, False]),
            "mle_regression": Cat([True, False]),
            "simple_differencing": Cat([True, False]),
            "enforce_stationarity": Cat([True, False]),
            "enforce_invertibility": Cat([True, False]),
            "hamilton_representation": Cat([True, False]),
            "concentrate_scale": Cat([True, False]),
            "use_exact_diffuse": Cat([True, False]),
        }

        # Drop order and seasonal_order params if specified by user
        if "order" in self._est_params:
            for p in self._order:
                dist.pop(p)
        if "seasonal_order" in self._est_params:
            for p in self._s_order:
                dist.pop(p)

        return dist


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

    In cases where ExponentialSmoothing results in a constant forecast,
    the theta forecaster will revert to predicting the SES constant plus
    a linear trend derived from the training data.

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

    The Vector Autoregressive (VAR) model is a type of multivariate
    time series model used for analyzing and forecasting the joint
    behavior of multiple time series variables. In a VAR model, each
    variable in the system is modeled as a linear combination of its
    past values as well as the past values of all other variables in
    the system. This allows for capturing the interdependencies and
    dynamic relationships among the variables over time.

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
    """Vector Autoregressive Moving-Average with exogenous variables.

    VARMAX is an extension of the [VAR][] model that incorporates not
    only lagged values of the endogenous variables, but also includes
    exogenous variables. This allows VARMAX models to capture both the
    interdependencies among multiple time series variables and the
    influence of external factors.

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
        if all(p in params for p in self._order):
            params["order"] = [params.pop(p) for p in self._order]

        # Drop order params
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

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = {
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
        }

        # Drop order params if specified by user
        if "order" in self._est_params:
            for p in self._order:
                dist.pop(p)

        return dist

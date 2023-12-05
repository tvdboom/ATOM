"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing all time series models.

"""

from __future__ import annotations

from typing import Any, ClassVar

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution as Cat
from optuna.distributions import IntDistribution as Int
from optuna.trial import Trial

from atom.basemodel import ForecastModel


class ARIMA(ForecastModel):
    """Autoregressive Integrated Moving Average Model.

    Seasonal ARIMA models and exogeneous input is supported, hence this
    estimator is capable of fitting SARIMA, ARIMAX, and SARIMAX.

    An ARIMA model, is a generalization of an autoregressive moving
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
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.arima"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "ARIMA"}

    _order = ("p", "d", "q")
    _sorder = ("P", "D", "Q", "S")

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
        if self._get_param("S", params) == 0:
            for p in self._sorder:
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
            params["order"] = tuple(params.pop(p) for p in self._order)
        if all(p in params for p in self._sorder):
            params["seasonal_order"] = tuple(params.pop(p) for p in self._sorder)

        return params

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        methods = ["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]

        dist = {
            "p": Int(0, 2),
            "d": Int(0, 1),
            "q": Int(0, 2),
            "P": Int(0, 2),
            "D": Int(0, 1),
            "Q": Int(0, 2),
            "S": Cat([0, 4, 6, 7, 12]),
            "method": Cat(methods),
            "maxiter": Int(50, 200, step=10),
            "with_intercept": Cat([True, False]),
        }

        # Drop order and seasonal_order params if specified by user
        if "order" in self._est_params:
            for p in self._order:
                dist.pop(p)
        if "seasonal_order" in self._est_params:
            for p in self._sorder:
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
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
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


class ExponentialSmoothing(ForecastModel):
    """Exponential Smoothing forecaster.

    Holt-Winters exponential smoothing forecaster. The default settings
    use simple exponential smoothing, without trend and seasonality
    components.

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
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.exp_smoothing"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "ExponentialSmoothing"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        methods = ["L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr", "bh", "ls"]

        return {
            "trend": Cat(["add", "mul", None]),
            "damped_trend": Cat([True, False]),
            "seasonal": Cat(["add", "mul", None]),
            "sp": Cat([4, 6, 7, 12, None]),
            "use_boxcox": Cat([True, False]),
            "initialization_method": Cat(["estimated", "heuristic"]),
            "method": Cat(methods),
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
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
    supports_engines = ("sktime",)

    _module = "sktime.forecasting.ets"
    _estimators: ClassVar[dict[str, str]] = {"forecast": "AutoETS"}

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
            "sp": Cat([1, 4, 6, 7, 12]),
            "initialization_method": Cat(["estimated", "heuristic"]),
            "maxiter": Int(500, 2000, step=100),
            "auto": Cat([True, False]),
            "information_criterion": Cat(["aic", "bic", "aicc"]),
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
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
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
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
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

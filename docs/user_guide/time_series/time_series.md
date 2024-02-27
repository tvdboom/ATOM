# Time series
-------------

Time series applies machine learning techniques to sequential data, where
observations are ordered over time. This approach is crucial for predicting
future values or events and finds applications in finance, healthcare, weather
forecasting, and more. ATOM support two time series tasks: univariate forecast
and multivariate forecast.

!!! example
    See our time series examples for [univariate][example-univariate-forecast]
    and [multivariate][example-multivariate-forecast] forecast.

<br>

## Exogenous variables

Exogenous variables are external factors that can influence the target variable
and, unlike endogenous variables, are not part of the time series being analyzed.

Incorporating exogenous variables into time series models helps capture additional
information that may impact the observed patterns. This inclusion allows for a
more comprehensive understanding of the underlying dynamics and can lead to more
accurate predictions.

Exogenous variables are added to atom with the `X` variable. When no exogenous
variables are provided, `atom.X` returns an empty dataframe. Note that not all
models make use of exogenous variables. Read more [here][model-selection] about
how to check specific model characteristics.

<br>

## Seasonality

Seasonality refers to the recurring patterns that repeat at regular intervals
over time, often corresponding to specific time periods, such as days, weeks,
or months, and can significantly influence the observed data.

Add seasonality to atom using the [`sp`][atomforecaster-sp] parameter or attribute.
You can add a single value for single seasonality or a sequence of values for
multiple seasonalities. If you don't know the seasonality a priori, you can use
the [`get_seasonal_period`][atomforecaster-get_seasonal_period] method to
automatically detect the seasonality, e.g. `#!python atom.sp = atom.get_seasonal_period()`
or directly from the constructor `#!python atom = ATOMForecaster(y, sp="infer")`.

The majority of models only support one seasonal period. If more than one period
is defined, such models only use the first one. Read [here][model-selection] how
to check which models support multiple seasonality.

!!! info
    In a [multivariate][] setting, the same period is used for all target columns.

In addition to the period, it's possible to further tune the seasonality by
specifying the trend and seasonal models. In an `additive` model, the components
are added together. It implies that the effect of one component does not depend
on the level of the other components. In a `multiplicative` model, the components
are multiplied together. This suggests that the effect of one component is
proportional to the level of the other components.

Specify the trend and/or seasonal models providing the `sp` parameter (or attribute)
with a dictionary, e.g., `#!python atom.sp = {"sp": 12, "seasonal_model": "multiplicative"}`.
Both the `seasonal_trend` and `seasonal_model` values default to `additive`.

<br>

## Forecasting with regressors

All of ATOM's [regressors][predefined-models] can also be used in forecasting
tasks. Simply select the regressor like any other model, e.g.,
`#!python atom.run(models="RF")` to use a [RandomForest][] model.

The regressor is automatically converted to a forecaster, based on reduction
to tabular or time-series regression. During fitting, a sliding-window approach
is used to first transform the time series into tabular or panel data, which is
then used to fit a tabular or time-series regression estimator. During prediction,
the last available data is used as input to the fitted regression estimator to
generate forecasts.

See below a graphical representation of the reduction logic using the following
symbols:

- y: forecast target.
- x: past values of y that are used as features (X) to forecast y.
- *:observations, past or future, neither part of the window nor forecast.

Assume we have the following training data (15 observations):

|------------------------------|
| * * * * * * * * * * * * * * *|
|------------------------------|

The reducer targets the first data point after the window, irrespective of the
forecasting horizons requested. In the example, the following five windows are
created:

|------------------------------|
| x x x x x x x x x x y * * * *|
| * x x x x x x x x x x y * * *|
| * * x x x x x x x x x x y * *|
| * * * x x x x x x x x x x y *|
| * * * * x x x x x x x x x x y|
|------------------------------|

!!! warning
    Regressor forecasters do not support in-sample predictions. Scores on the
    training set always return `NaN`.

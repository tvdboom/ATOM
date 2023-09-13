# AutoARIMA
-----------

:: atom.models:AutoARIMA
    :: tags
    :: description
    :: see also

<br>

## Example

:: examples

<br><br>

## Hyperparameters

:: hyperparameters

<br><br>

## Attributes

### Data attributes

:: table:
    - attributes:
        from_docstring: False
        include:
            - pipeline
            - atom.branch:Branch.mapping
            - dataset
            - train
            - test
            - X
            - y
            - X_train
            - y_train
            - X_test
            - atom.branch:Branch.y_test
            - X_holdout
            - y_holdout
            - atom.branch:Branch.shape
            - atom.branch:Branch.columns
            - atom.branch:Branch.n_columns
            - atom.branch:Branch.features
            - atom.branch:Branch.n_features
            - atom.branch:Branch.target

<br>

### Utility attributes

:: table:
    - attributes:
        from_docstring: False
        include:
            - name
            - study
            - trials
            - best_trial
            - best_params
            - score_ht
            - time_ht
            - estimator
            - score_train
            - score_test
            - score_holdout
            - time_fit
            - bootstrap
            - score_bootstrap
            - time_bootstrap
            - time
            - feature_importance
            - results

<br>

### Prediction attributes

The [prediction attributes][] are not calculated until the attribute
is called for the first time. This mechanism avoids having to calculate
attributes that are never used, saving time and memory.

:: table:
    - attributes:
        from_docstring: False
        include:
            - atom.basemodel:ForecastModel.predict_train
            - atom.basemodel:ForecastModel.predict_test
            - atom.basemodel:ForecastModel.predict_holdout
            - atom.basemodel:ForecastModel.predict_interval_train
            - atom.basemodel:ForecastModel.predict_interval_test
            - atom.basemodel:ForecastModel.predict_interval_holdout
            - atom.basemodel:ForecastModel.predict_proba_train
            - atom.basemodel:ForecastModel.predict_proba_test
            - atom.basemodel:ForecastModel.predict_proba_holdout
            - atom.basemodel:ForecastModel.predict_quantiles_train
            - atom.basemodel:ForecastModel.predict_quantiles_test
            - atom.basemodel:ForecastModel.predict_quantiles_holdout
            - atom.basemodel:ForecastModel.predict_residuals_train
            - atom.basemodel:ForecastModel.predict_residuals_test
            - atom.basemodel:ForecastModel.predict_residuals_holdout
            - atom.basemodel:ForecastModel.predict_var_train
            - atom.basemodel:ForecastModel.predict_var_test
            - atom.basemodel:ForecastModel.predict_var_holdout

<br><br>

## Methods

The [plots][available-plots] can be called directly from the model.
The remaining utility methods can be found hereunder.

:: methods:
    toc_only: False
    exclude:
        - plot_.*

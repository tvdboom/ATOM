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
            - atom.data:Branch.mapping
            - dataset
            - train
            - test
            - X
            - y
            - X_train
            - y_train
            - X_test
            - atom.data:Branch.y_test
            - X_holdout
            - y_holdout
            - shape
            - columns
            - n_columns
            - features
            - n_features
            - atom.data:Branch.target

<br>

### Utility attributes

:: table:
    - attributes:
        from_docstring: False
        include:
            - name
            - run
            - study
            - trials
            - best_trial
            - best_params
            - estimator
            - bootstrap
            - results
            - feature_importance

<br><br>

## Methods

The [plots][available-plots] can be called directly from the model.
The remaining utility methods can be found hereunder.

:: methods:
    toc_only: False
    exclude:
        - plot_.*

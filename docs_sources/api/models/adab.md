# AdaBoost
----------

:: atom.models:AdaBoost
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
            - atom.branch:Branch.pipeline
            - atom.branch:Branch.mapping
            - atom.branch:Branch.dataset
            - atom.branch:Branch.train
            - atom.branch:Branch.test
            - atom.branch:Branch.X
            - atom.branch:Branch.y
            - atom.branch:Branch.X_train
            - atom.branch:Branch.y_train
            - atom.branch:Branch.X_test
            - atom.branch:Branch.y_test
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
            - atom.basemodel:BaseModel.decision_function_train
            - atom.basemodel:BaseModel.decision_function_test
            - atom.basemodel:BaseModel.decision_function_holdout
            - atom.basemodel:BaseModel.predict_train
            - atom.basemodel:BaseModel.predict_test
            - atom.basemodel:BaseModel.predict_holdout
            - atom.basemodel:BaseModel.predict_log_proba_train
            - atom.basemodel:BaseModel.predict_log_proba_test
            - atom.basemodel:BaseModel.predict_log_proba_holdout
            - atom.basemodel:BaseModel.predict_proba_train
            - atom.basemodel:BaseModel.predict_proba_test
            - atom.basemodel:BaseModel.predict_proba_holdout


<br><br>

## Methods

The [model plots][] and [prediction methods][] can be called directly
from the model. The remaining utility methods can be found hereunder.

:: methods:
    toc_only: False
    include:
        - bootstrapping
        - calibrate
        - clear
        - create_app
        - create_dashboard
        - cross_validate
        - delete
        - evaluate
        - export_pipeline
        - fit
        - full_train
        - hyperparameter_tuning
        - inverse_transform
        - save_estimator
        - transform

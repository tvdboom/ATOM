# SuccessiveHalvingRegressor
----------------------------

:: atom.training:SuccessiveHalvingRegressor
    :: signature
    :: head
    :: table:
        - parameters
    :: see also

<br>

## Example

:: examples

<br>

## Attributes

### Data attributes

The data attributes are used to access the dataset and its properties.
Updating the dataset will automatically update the response of these
attributes accordingly.

:: table:
    - attributes:
        from_docstring: False
        include:
            - atom.data:Branch.dataset
            - atom.data:Branch.train
            - atom.data:Branch.test
            - atom.data:Branch.X
            - atom.data:Branch.y
            - holdout
            - atom.data:Branch.X_train
            - atom.data:Branch.y_train
            - atom.data:Branch.X_test
            - atom.data:Branch.y_test
            - atom.data:Branch.shape
            - atom.data:Branch.columns
            - atom.data:Branch.n_columns
            - atom.data:Branch.features
            - atom.data:Branch.n_features
            - atom.data:Branch.target

<br>

### Utility attributes

The utility attributes are used to access information about the models
in the instance after [training][].

:: table:
    - attributes:
        from_docstring: False
        include:
            - models
            - metric
            - winners
            - winner
            - results

<br>

### Tracking attributes

The tracking attributes are used to customize what elements of the
experiment are tracked. Read more in the [user guide][tracking].

:: table:
    - attributes:
        from_docstring: False
        include:
            - log_ht
            - log_plots
            - log_data
            - log_pipeline

<br>

### Plot attributes

The plot attributes are used to customize the plot's aesthetics. Read
more in the [user guide][aesthetics].

:: table:
    - attributes:
        from_docstring: False
        include:
            - palette
            - title_fontsize
            - label_fontsize
            - tick_fontsize
            - line_width
            - marker_size

<br>

## Methods

Next to the [plotting][plots] methods, the class contains a variety
of methods to handle the data, run the training, and manage the pipeline.

:: methods:
    toc_only: False
    include:
        - available_models
        - canvas
        - clear
        - delete
        - evaluate
        - export_pipeline
        - get_class_weight
        - get_params
        - merge
        - update_layout
        - update_traces
        - reset_aesthetics
        - run
        - save
        - set_params
        - stacking
        - voting

# Frequently asked questions
----------------------------

Here we try to give answers to some questions that have popped up
regularly. If you have any other questions, don't hesitate to create
a new [discussion](https://github.com/tvdboom/ATOM/discussions) or post
them on the [Slack channel](https://join.slack.com/t/atom-alm7229/shared_invite/zt-upd8uc0z-LL63MzBWxFf5tVWOGCBY5g)! 

??? faq "How does ATOM relate to AutoML?"
    There is, indeed, a text editor with the same name and a similar logo as
    this package. Is this a shameless copy? No. When I started the project,
    I didn't know about the text editor, and it doesn't require much thinking
    to come up with the idea of replacing the letter O of the word atom with
    the image of an atom.

??? faq "Is this package related to the Atom text editor?"
    ATOM is not an AutoML tool since it does not automate the search for
    an optimal pipeline like well known AutoML tools such as
    [auto-sklearn](https://automl.github.io/auto-sklearn/master/) or
    [EvalML](https://evalml.alteryx.com/en/stable/) do. Instead, ATOM helps
    the user find the optimal pipeline himself. One of the goals of this
    package is to help data scientists produce explainable pipelines, and
    using an AutoML black box function would impede that. That said, it is
    possible to integrate a EvalML pipeline with atom through the
    [automl](../API/ATOM/atomclassifier/#automl) method.

??? faq "Is it possible to run deep learning models?"
    Yes. Deep learning models can be added as custom models to the pipeline
    as long as they follow [sklearn's API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).
    For more information, see the [deep learning][deep-learning] section
    of the user guide.

??? faq "Can I run atom's methods on just a subset of the columns?"
    Yes, all [data cleaning][data-cleaning] and [feature engineering]
    [feature-engineering] methods accept a `columns` parameter to only
    transform the selected features. For example, to only impute the
    numerical columns in the dataset we could type `atom.impute(strat_num="mean",
    columns=atom.numerical)`. The parameter accepts column names, column
    indices, dtypes or a slice object.

??? faq "How can I compare the same model on different datasets?"
    In many occasions you might want to test how a model performs on datasets
    processed with different pipelines. For this, atom has the [branch system]
    [branches]. Create a new branch for every new pipeline you want to test
    and use the plot methods to compare all models, independent of the branch
    it was trained on.

??? faq "Can I train models through atom using a GPU?"
    Yes. Refer to the [user guide][gpu-acceleration] to see what algorithms
    and models have a GPU implementation. Be aware that it requires additional
    software and hardware dependencies.

??? faq "How are numerical and categorical columns differentiated?"
    The columns are separated using a dataframe's [select_dtypes](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html)
    method. Numerical columns are selected using `include="number"`
    whereas categorical columns are selected using `exclude="number"`.

??? faq "Can I run unsupervised learning pipelines?"
    No. As for now, ATOM only supports supervised machine learning pipelines.
    However, various unsupervised algorithms can be chosen as strategy in the
    [Pruner][pruner] class to detect and remove outliers from the dataset.

??? faq "Is there a way to plot multiple models in the same shap plot?"
    No. Unfortunately, there is no way to plot multiple models in the same
    [shap plot][shap] since the plots are made by the [shap](https://github.com/slundberg/shap)
    package and passed as `matplotlib.axes` objects to atom. This means
    that it's not within the reach of this package to implement such a
    utility.

??? faq "Can I merge a sklearn pipeline with atom?"
    Yes. Like any other transformer, it is possible to add a sklearn
    pipeline to atom using the [add](../API/ATOM/atomclassifier/#add)
    method. Every transformer in the pipeline is merged independently.
    The pipeline is not allowed to end with a model since atom manages
    its own models. If that is the case, add the pipeline using
    `atom.add(pipeline[:-1])`.

??? faq "Is it possible to initialize atom with an existing train and test set?"
    Yes. If you already have a separated train and test set you can
    initialize atom in two ways:

    * `atom = ATOMClassifier(train, test)`
    * `atom = ATOMClassifier((X_train, y_train), (X_test, y_test))`

    Make sure the train and test size have the same number of columns! If
    atom is initialized in any of these two ways, the `test_size` parameter
    is ignored.

??? faq "Can I train the models using cross-validation?"
    Applying cross-validation means transforming every step of the pipeline
    multiple times, each with different results. Doing this would prevent
    ATOM from being able to show the transformation results after every
    pre-processing step, which means losing the ability to inspect how a
    transformer changed the dataset. For this reason, it is not possible to
    apply cross-validation until after a model has been trained. After a
    model has been trained, the pipeline is defined, and cross-validation
    can be applied using the [cross_validate](../API/models/gnb/#cross-validate)
    method. See [here][multi-metric-runs] an example using cross-validation.

??? faq "Is there a way to process datetime features?"
    Yes, the [FeatureExtractor][featureextractor] class can automatically
    extract useful features (day, month, year, etc...) from datetime columns.
    The extracted features are always encoded to numerical values, so they
    can be fed directly to a model.

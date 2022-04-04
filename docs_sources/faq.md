# Frequently asked questions
----------------------------

Here we try to give answers to some questions that have popped up
regularly. If you have any other questions, don't hesitate to create
a new [discussion](https://github.com/tvdboom/ATOM/discussions) or post
them on the [slack channel](https://join.slack.com/t/atom-alm7229/shared_invite/zt-upd8uc0z-LL63MzBWxFf5tVWOGCBY5g)! 


1. [Is this package related to the text editor?](#q1)
2. [How does ATOM relate to AutoML?](#q2)
3. [Is it possible to run deep learning models?](#q3)
4. [Can I run atom's methods on just a subset of the columns?](#q4)
5. [How can I compare the same model on different datasets?](#q5)
6. [Can I train models through atom using a GPU?](#q6)
7. [How are numerical and categorical columns differentiated?](#q7)
8. [Can I run unsupervised learning pipelines?](#q8)
9. [Is there a way to plot multiple models in the same shap plot?](#q9)
10. [Can I merge a sklearn pipeline with atom?](#q10)
11. [Is it possible to initialize atom with an existing train and test set?](#q11)
12. [Can I train the models using cross-validation?](#q12)
13. [Is there a way to process datetime features?](#q13)


<br>

------

<a name="q1"></a>
### 1. Is this package related to the text editor?

There is, indeed, a text editor with the same name and a similar logo as this
package. Is this a shameless copy? No. When I started the project, I didn't
know about the text editor, and it doesn't require much thinking to come up
with the idea of replacing the letter O of the word atom with the image of
an atom.

<br>

<a name="q2"></a>
### 2. How does ATOM relate to AutoML?

ATOM is not an AutoML tool since it does not automate the search for
an optimal pipeline like well known AutoML tools such as
[auto-sklearn](https://automl.github.io/auto-sklearn/master/) or
[TPOT](http://epistasislab.github.io/tpot/) do. Instead, ATOM helps
the user find the optimal pipeline himself. One of the goals of this
package is to help data scientists produce explainable pipelines, and
using an AutoML black box function would impede that. That said, it is
possible to integrate a TPOT pipeline with atom through the
[automl](../API/ATOM/atomclassifier/#automl) method.

<br>

<a name="q3"></a>
### 3. Is it possible to run deep learning models?

Yes. Deep learning models can be added as custom models to the pipeline
as long as they follow [sklearn's API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).
If the dataset is 2-dimensional, everything should work normally. If
the dataset has more than 2 dimensions, often the case for images, only
a subset of atom's methods will work. For more information, see the
[deep learning](../user_guide/models/#deep-learning) section of the user guide.

<br>

<a name="q4"></a>
### 4. Can I run atom's methods on just a subset of the columns?

Yes, all [data cleaning](../user_guide/data_cleaning) and
[feature engineering](../user_guide/feature_engineering) methods accept
a `columns` parameter to only transform the selected features. For example,
to only impute the numerical columns in the dataset we could type
`atom.impute(strat_num="mean", columns=atom.numerical)`. The parameter
accepts column names, column indices, dtypes or a slice object.

<br>

<a name="q5"></a>
### 5. How can I compare the same model on different datasets?

In many occasions you might want to test how a model performs on datasets
processed with different pipelines. For this, atom has the [branch system](../user_guide/data_management/#branches).
Create a new branch for every new pipeline you want to test and use the plot
methods to compare all models, independent of the branch it was trained on.

<br>

<a name="q6"></a>
### 6. Can I train models through atom using a GPU?

Yes. Refer to the [user guide](../user_guide/gpu) to see what algorithms
and models have a GPU implementation. Be aware that it requires additional
software and hardware dependencies.

<br>

<a name="q7"></a>
### 7. How are numerical and categorical columns differentiated?

The columns are separated using a dataframe's [select_dtypes](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html)
method. Numerical columns are selected using `include="number"`
whereas categorical columns are selected using `exclude="number"`.

<br>

<a name="q8"></a>
### 8. Can I run unsupervised learning pipelines?

No. As for now, ATOM only supports supervised machine learning pipelines.
However, various unsupervised algorithms can be chosen as strategy in the
[Pruner](../API/data_cleaning/pruner) class to detect and remove outliers
from the dataset.

<br>

<a name="q9"></a>
### 9. Is there a way to plot multiple models in the same shap plot?

No. Unfortunately, there is no way to plot multiple models in the same
[shap plot](../user_guide/plots/#shap) since the plots are made by the [shap](https://github.com/slundberg/shap)
package and passed as matplotlib.axes objects to atom. This means
that it's not within the reach of this package to implement such a
utility.

<br>

<a name="q10"></a>
### 10. Can I merge a sklearn pipeline with atom?

Yes. Like any other transformer, it is possible to add a sklearn
pipeline to atom using the [add](../API/ATOM/atomclassifier/#add)
method. Every transformer in the pipeline is merged
independently. The pipeline is not allowed to end with a model
since atom manages its own models. If that is the case, add the
pipeline using `atom.add(pipeline[:-1])`.

<br>

<a name="q11"></a>
### 11. Is it possible to initialize atom with an existing train and test set?

Yes. If you already have a separated train and test set you can initialize
atom in two ways:

* `atom = ATOMClassifier(train, test)`
* `atom = ATOMClassifier((X_train, y_train), (X_test, y_test))`

Make sure the train and test size have the same number of columns! If
atom is initialized in any of these two ways, the `test_size` parameter
is ignored.

<br>

<a name="q12"></a>
### 12. Can I train the models using cross-validation?
It is not possible to train models using cross-validation, but for a
good reason. Applying cross-validation would mean transforming every
step of the pipeline multiple times, each with different results. This
would prevent ATOM from being able to show the transformation results
after every pre-processing step, which means losing the ability to inspect
how a transformer changed the dataset. This makes cross-validation an
inappropriate technique for the purpose of exploration.

So why not use cross-validation only to train and evaluate the models,
instead of applying it to the whole pipeline? Cross-validating only the
models would make no sense here. If we use the complete dataset for
that (both the train and test set), we would be evaluating the models
on data that was used to fit the transformers. This implies data leakage
and can severely bias the results towards specific transformers. On the
other hand, using only the training set beats the point of applying
cross-validation in the first place, since we can train the model on the
complete training set and evaluate the results on the independent test
set. The only reason of doing cross-validation would be to get an idea
of the robustness of the model. This can also be achieves using
[bootstrapping](../user_guide/training/#bootstrapping). That said, ideally
we would cross-validate the entire pipeline using the entire dataset.
This can be done using a trainer's [cross_validate](../API/models/gnb/#cross-validate)
method, but for the reason just explained above, the method only outputs
the final metric results.

<br>

<a name="q13"></a>
### 13. Is there a way to process datetime features?
Yes, the [FeatureExtractor](../API/feature_engineering/feature_extractor)
class can automatically extract useful features (day, month, year, etc...)
from datetime columns. The extracted features are always encoded to numerical
values, so they can be fed directly to a model.

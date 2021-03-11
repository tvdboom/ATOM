# Frequently asked questions
----------------------------

* [There already is an atom text editor. Does this has anything to do with that?](#q1)
* [How does ATOM relate to AutoML?](#q2)
* [Is it possible to run deep learning models with ATOM?](#q3)
* [Can I run atom's methods on just a subset of the columns?](#q4)
* [How can I compare the same model on different datasets?](#q5)
* [Can I train models through atom using a GPU?](#q6)

<br>

------

<br>

<a name="q1"></a>
### There already is an atom text editor. Does this has anything to do with that?

There is, indeed, a text editor with the same name and a similar logo. Is this
a shameless copy? No. When I started the project, I didn't know about the text
editor, and it doesn't require much thinking to come up with the idea of replacing
the letter O of the word atom with the image of an atom.

<br>

<a name="q2"></a>
### How does ATOM relate to AutoML?

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
### Is it possible to run deep learning models with ATOM?

Yes. Deep learning models can be added as custom models to the pipeline
as long as they follow [sklearn's API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).
If the dataset is 2-dimensional, everything should work normally. If
the dataset has more than 2 dimensions (referred in the documentation as
deep learning datasets, often the case for images or text embeddings),
only a subset of atom's methods will work. For more information, see
the [deep learning](../user_guide/#deep-learning) section of the user guide.

<br>

<a name="q4"></a>
### Can I run atom's methods on just a subset of the columns?

Yes, all [data cleaning](../user_guide/#data-cleaning) and
[feature engineering](../user_guide/#feature-engineering) methods accept
a `columns` parameter to only transform the selected features. For example,
to only impute the numerical columns in the dataset we could type
`atom.impute(strat_num="mean", columns=atom.numerical)`. The parameter
accepts column names, column indices or a slice object.

<br>

<a name="q5"></a>
### How can I compare the same model on different datasets?

In many occasions you might want to test how a model performs on datasets
processed with different pipelines. For this, atom has the [branch system](../user_guide/#branches).
Create a new branch for every new pipeline you want to test and use the plot
methods to compare all models, independent of the branch it was trained on.

<br>

<a name="q6"></a>
### Can I train models through atom using a GPU?

ATOM doesn't fit the models himself. The underlying models' package does. 
Since the majority of predefined models are implemented through sklearn
and sklearn works on CPU only, they can not be trained on any GPU. If you
are using a custom model whose package, Keras for example, allows GPU
implementation and the settings or model parameters are tuned to do so, the
model will train on the GPU like it would do outside atom.

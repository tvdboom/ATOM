# Natural Language Processing
-----------------------------

Natural Language Processing (NLP) is the subfield of machine learning
that works with human language data. The nlp module contains four
classes that help to convert raw text to meaningful numeric values,
ready to be ingested by a model. ATOM uses the [nltk](https://www.nltk.org/index.html)
library for the majority of its NLP processes.

The text documents are expected to be provided in a column of the dataframe
named `corpus` (the name is case-insensitive). Only the corpus is changed
by the transformers, leaving the rest of the columns as is. This mechanism
allows atom to combine datasets containing a text corpus with other non-text
features. If an array is provided as input, it should consist of only one
feature containing the text (one document per row). ATOM will then
automatically convert the array to a dataframe with the desired column name.
Documents are expected to be strings or sequences of words. Click
[here][example-nlp] for an example using text data.

!!! note
    All of atom's NLP methods automatically adopt the relevant transformer
    attributes (`verbose`, `logger`) from atom. A different choice can be
    added as parameter to the method call, e.g. `#!python atom.tokenize(verbose=0)`.

!!! info
    ATOM doesn't do topic modelling! The module's goal is to help process
    text documents into features that can be used for supervised learning.

<br>

## Text cleaning

Text data is rarely clean. Whether it's scraped from a website or inferred
from paper documents, it's always populated with irrelevant information for
the model, such as email addresses, HTML tags, numbers or punctuation marks.
Use the [TextCleaner][] class to clean the corpus from such noise. It can be
accessed from atom through the [textclean][atomclassifier-textclean] method.
Use the class' parameters to choose which transformations to perform. The
available steps are:

* Decode unicode characters to their ascii representations.
* Convert all characters to lower case.
* Drop email addresses from the text.
* Drop URL links from the text.
* Drop HTML tags from the text.
* Drop emojis from the text.
* Drop numbers from the text.
* Drop punctuations from the text.


<br>

## Tokenization

Some text processing algorithms, like stemming or lemmatization, require the
corpus to be made out of tokens, instead of strings, in order to know what to
consider as words. Tokenization is used to achieve this. It separates every
document into a sequence of smaller units. In this case, the words.

Sometimes, words have a different meaning on their own than when combined
with adjacent words. For example, the word `new` has a completely different
meaning when the word `york` is directly after it than when it's not. These
combinations of two words are called bigrams. When there are three words,
they are called trigrams, and with four words quadgrams.

The [Tokenizer][] class converts a document into a sequence of words, and
can create the most frequent bigrams, trigrams and quadgrams. It can be
accessed from atom through the [tokenize][atomclassifier-tokenize] method.


<br>

## Text Normalization

Normalization for texts is a process that converts a list of words to a
more uniform standard. This is useful to reduce the amount of different
information that the computer has to deal with, and therefore improves
efficiency. The goal of normalization techniques like stemming and
lemmatization is to reduce inflectional and related forms of a word
to a common base form.

Normalize the words in the corpus using the [TextNormalizer][] class.
It can be accessed from atom through the [textnormalize][atomclassifier-textnormalize]
method.


<br>

## Vectorization

Text data cannot be fed directly to the algorithms themselves, as most
of them expect numerical feature vectors with a fixed size, rather than
words in the text documents with variable length. Vectorization is the
general process of turning a collection of text documents into numerical
feature vectors. You can apply it to the corpus using the [Vectorizer][]
class. It can be accessed from atom through the [vectorize][atomclassifier-vectorize]
method.

!!! info
    All strategies can utilize GPU speed-up. Click [here][gpu-acceleration]
    for further information about GPU acceleration.

<br style="display: block; margin-top: 2em; content: ' '">

**Bag of Words**<br>
The Bag of Words (BOW) strategy applies tokenization, counting and
normalization to the corpus. Documents are described by word occurrences
while completely ignoring the relative position information of the words
in the document. The created columns are named with the words they are
embedding with the prefix `corpus_`. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation).

<br style="display: block; margin-top: 2em; content: ' '">

**TF-IDF**<br>
In a large text corpus, some words will be very present (e.g. “the”,
“a”, “is” in English), hence carrying very little meaningful information
about the actual contents of the document. If we were to feed the direct
count data directly to a classifier, those very frequent terms would
shadow the frequencies of rarer, yet more interesting, terms. Use the
TF-IDF strategy to re-weight the count features into floating point values.
The created columns are named with the words they are embedding with the
prefix `corpus_`. Read more in sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting).

<br style="display: block; margin-top: 2em; content: ' '">

**Hashing**<br>
The larger the corpus, the larger the vocabulary will grow and thus
increasing the number of features and memory use. Use the Hashing
strategy to hash the words to a specified number of features. The
created features are named `hash0`, `hash1`, etc... Read more in
sklearn's [documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#vectorizing-a-large-text-corpus-with-the-hashing-trick).

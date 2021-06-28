# Tokenizer
-----------

<div style="font-size:20px">
<em>class</em> atom.nlp.<strong style="color:#008AB8">Tokenizer</strong>(bigram_freq=None,
trigram_freq=None, quadgram_freq=None, verbose=0, logger=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/nlp.py#L284">[source]</a>
</span>
</div>

Convert documents into sequences of words. Additionally, create n-grams
(represented by words united with underscores, e.g. "New_York") based
on their frequency in the corpus. The transformations are applied on
the column named `Corpus`. If there is no column with that name, an
exception is raised. This class can be accessed from atom through the
[tokenize](../../ATOM/atomclassifier/#tokenize) method. Read more
in the [user guide](../../../user_guide/nlp/#tokenization).

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>bigram_freq: int, float or None, optional (default=None)</strong><br>
Frequency threshold for bigram creation.
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: Don't create any bigrams.</li>
<li>If int: Minimum number of occurrences to make a bigram.</li>
<li>If float: Minimum frequency fraction to make a bigram.</li>
</ul>
<strong>trigram_freq: int, float or None, optional (default=None)</strong><br>
Frequency threshold for trigram creation.
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: Don't create any trigrams.</li>
<li>If int: Minimum number of occurrences to make a trigram.</li>
<li>If float: Minimum frequency fraction to make a trigram.</li>
</ul>
<strong>quadgram_freq: int, float or None, optional (default=None)</strong><br>
Frequency threshold for quadgram creation.
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: Don't create any quadgrams.</li>
<li>If int: Minimum number of occurrences to make a quadgram.</li>
<li>If float: Minimum frequency fraction to make a quadgram.</li>
</ul>
<strong>verbose: int, optional (default=0)</strong><br>
Verbosity level of the class. Possible values are:
<ul style="line-height:1.2em;margin-top:5px">
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
<strong>logger: str, Logger or None, optional (default=None)</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: Doesn't save a logging file.</li>
<li>If str: Name of the log file. Use "auto" for automatic naming.</li>
<li>Else: Python <code>logging.Logger</code> instance.</li>
</ul>
</td>
</tr>
</table>

<br>


## Attributes

<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>bigrams: pd.DataFrame</strong><br>
Created bigrams and their frequencies.
</p>
<p>
<strong>trigrams: pd.DataFrame</strong><br>
Created trigrams and their frequencies.
</p>
<p>
<strong>quadgrams: pd.DataFrame</strong><br>
Created quadgrams and their frequencies.
</p>
</td>
</tr>
</table>
<br>


## Methods

<table style="font-size:16px">
<tr>
<td><a href="#fit-transform">fit_transform</a></td>
<td>Same as transform.</td>
</tr>

<tr>
<td><a href="#get-params">get_params</a></td>
<td>Get parameters for this estimator.</td>
</tr>

<tr>
<td><a href="#log">log</a></td>
<td>Write information to the logger and print to stdout.</td>
</tr>

<tr>
<td><a href="#save">save</a></td>
<td>Save the instance to a pickle file.</td>
</tr>

<tr>
<td><a href="#set-params">set_params</a></td>
<td>Set the parameters of this estimator.</td>
</tr>

<tr>
<td><a href="#transform">transform</a></td>
<td>Transform the text.</td>
</tr>
</table>
<br>


<a name="fit-transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L74">[source]</a>
</span>
</div>
Tokenize the text.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong><br>
Feature set with shape=(n_samples, n_features). If X is
not a pd.DataFrame, it should be composed of a single
feature containing the text documents.
</p>
<strong>y: int, str, sequence or None, optional (default=None)</strong><br>
Does nothing. Implemented for continuity of the API.
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>X: pd.DataFrame</strong><br>
Transformed corpus.
</p>
</td>
</tr>
</table>
<br />


<a name="get-params"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">get_params</strong>(deep=True)
<span style="float:right">
<a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L189">[source]</a>
</span>
</div>
Get parameters for this estimator.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>deep: bool, optional (default=True)</strong><br>
If True, will return the parameters for this estimator and contained subobjects that are estimators.
</p>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>params: dict</strong><br>
Dictionary of the parameter names mapped to their values.
</td>
</tr>
</table>
<br />


<a name="log"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L348">[source]</a>
</span>
</div>
Write a message to the logger and print it to stdout.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>msg: str</strong><br>
Message to write to the logger and print to stdout.
</p>
<p>
<strong>level: int, optional (default=0)</strong><br>
Minimum verbosity level to print the message.
</p>
</td>
</tr>
</table>
<br />


<a name="save"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">save</strong>(filename="auto")
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L369">[source]</a>
</span>
</div>
Save the instance to a pickle file.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>filename: str, optional (default="auto")</strong><br>
Name of the file. Use "auto" for automatic naming.
</td>
</tr>
</table>
<br>


<a name="set-params"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">set_params</strong>(**params)
<span style="float:right">
<a href="https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/base.py#L221">[source]</a>
</span>
</div>
Set the parameters of this estimator.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>**params: dict</strong><br>
Estimator parameters.
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>self: Tokenizer</strong><br>
Estimator instance.
</td>
</tr>
</table>
<br />


<a name="transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/nlp.py#L355">[source]</a>
</span>
</div>
Tokenize the text.
<table style="font-size:16px">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>X: dict, list, tuple, np.ndarray or pd.DataFrame</strong><br>
Feature set with shape=(n_samples, n_features). If X is
not a pd.DataFrame, it should be composed of a single
feature containing the text documents.
</p>
<strong>y: int, str, sequence or None, optional (default=None)</strong><br>
Does nothing. Implemented for continuity of the API.
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<p>
<strong>X: pd.DataFrame</strong><br>
Transformed corpus.
</p>
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X, y)
atom.tokenize(bigram_freq=0.01)
```
or
```python
from atom.nlp import Tokenizer

tokenizer = Tokenizer(bigram_freq=0.01)
X = tokenizer.transform(X)
```
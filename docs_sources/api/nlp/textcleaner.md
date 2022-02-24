# TextCleaner
-------------

<div style="font-size:20px">
<em>class</em> atom.nlp.<strong style="color:#008AB8">TextCleaner</strong>(decode=True,
lower_case=True, drop_email=True, regex_email=None, drop_url=True,
regex_url=None, drop_html=True, regex_html=None, drop_emoji,
regex_emoji=None, drop_number=True, regex_number=None,
drop_punctuation=True, verbose=0, logger=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/nlp.py#L41">[source]</a>
</span>
</div>

Applies standard text cleaning to the corpus. Transformations include
normalizing characters and dropping noise from the text (emails, HTML
tags, URLs, etc...). The transformations are applied on the column
named `corpus`, in the same order the parameters are presented. If
there is no column with that name, an exception is raised. This class
can be accessed from atom through the [textclean](../../ATOM/atomclassifier/#textclean)
method. Read more in the [user guide](../../../user_guide/nlp/#text-cleaning).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>decode: bool, optional (default=True)</strong><br>
Whether to decode unicode characters to their ascii representations.
</p>
<p>
<strong>lower_case: bool, optional (default=True)</strong><br>
Whether to convert all characters to lower case.
</p>
<p>
<strong>drop_email: bool, optional (default=True)</strong><br>
Whether to drop email addresses from the text.
</p>
<p>
<strong>regex_email: str, optional (default=None)</strong><br>
Regex used to search for email addresses. If None, it uses
<code>r"[\w.-]+@[\w-]+\.[\w.-]+"</code>.
</p>
<p>
<strong>drop_url: bool, optional (default=True)</strong><br>
Whether to drop URL links from the text.
</p>
<p>
<strong>regex_url: str, optional (default=None)</strong><br>
Regex used to search for URLs. If None, it uses
<code>r"https?://\S+|www\.\S+"</code>.
</p>
<p>
<strong>drop_html: bool, optional (default=True)</strong><br>
Whether to drop HTML tags from the text. This option is
particularly useful if the data was scraped from a website.
</p>
<p>
<strong>regex_html: str, optional (default=None)</strong><br>
Regex used to search for html tags. If None, it uses
<code>r"<.*?>"</code>.
</p>
<p>
<strong>drop_emoji: bool, optional (default=True)</strong><br>
Whether to drop emojis from the text.
</p>
<p>
<strong>regex_emoji: str, optional (default=None)</strong><br>
Regex used to search for emojis. If None, it uses
<code>r":[a-z_]+:"</code>.
</p>
<p>
<strong>drop_number: bool, optional (default=False)</strong><br>
Whether to drop numbers from the text.
</p>
<p>
<strong>regex_number: str, optional (default=None)</strong><br>
Regex used to search for numbers. If None, it uses <code>r"\b\d+\b"</code>.
Note that numbers adjacent to letters are not removed.
</p>
<p>
<strong>drop_punctuation: bool, optional (default=True)</strong><br>
Whether to drop punctuations from the text. Characters considered
punctuation are <code>!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~</code>.
</p>
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
<td width="20%" class="td_title" style="vertical-align:top"><strong>Attributes:</strong></td>
<td width="80%" class="td_params">
<strong>drops: pd.DataFrame</strong><br>
Encountered regex matches. The row indices correspond to
the document index from which the occurrence was dropped.
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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L77">[source]</a>
</span>
</div>
Apply text cleaning.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like</strong><br>
Feature set with shape=(n_samples, n_features). If X is
not a pd.DataFrame, it should be composed of a single
feature containing the text documents.
</p>
<strong>y: int, str, sequence or None, optional (default=None)</strong><br>
Does nothing. Implemented for continuity of the API.
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
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
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>deep: bool, optional (default=True)</strong><br>
If True, will return the parameters for this estimator and contained
subobjects that are estimators.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>dict</strong><br>
Parameter names mapped to their values.
</td>
</tr>
</table>
<br />


<a name="log"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">log</strong>(msg, level=0)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L525">[source]</a>
</span>
</div>
Write a message to the logger and print it to stdout.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L546">[source]</a>
</span>
</div>
Save the instance to a pickle file.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
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
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>**params: dict</strong><br>
Estimator parameters.
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>TextCleaner</strong><br>
Estimator instance.
</td>
</tr>
</table>
<br />


<a name="transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/nlp.py#L157">[source]</a>
</span>
</div>
Apply text cleaning.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like</strong><br>
Feature set with shape=(n_samples, n_features). If X is
not a pd.DataFrame, it should be composed of a single
feature containing the text documents.
</p>
<strong>y: int, str, sequence or None, optional (default=None)</strong><br>
Does nothing. Implemented for continuity of the API.
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
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
atom.textclean()
```
or
```python
from atom.nlp import TextCleaner

cleaner = TextCleaner()
X = cleaner.transform(X)
```
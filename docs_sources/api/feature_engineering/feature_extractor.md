# FeatureExtractor
------------------

<div style="font-size:20px">
<em>class</em> atom.feature_engineering.<strong style="color:#008AB8">FeatureExtractor</strong>(features=["day", "month", "year"],
fmt=None, encoding_type="ordinal", drop_columns=True, verbose=0, logger=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/feature_engineering.py#L43">[source]</a>
</span>
</div>

Create new features extracting datetime elements (day, month, year,
etc...) from the provided columns. Columns of dtype `datetime64` are
used as is. Categorical columns that can be successfully converted
to a datetime format (less than 30% NaT values after conversion)
are also used. This class can be accessed from atom through the
[feature_extraction](../../ATOM/atomclassifier/#feature-extraction)
method. Read more in the [user guide](../../../user_guide/feature_engineering/#extracting-datetime-features).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>features: str or sequence, default=["day", "month", "year"]</strong><br>
Features to create from the datetime columns. Note that
created features with zero variance (e.g. the feature hour
in a column that only contains dates) are ignored. Allowed
values are datetime attributes from `pandas.Series.dt`.
</p>
<p>
<strong>fmt: str, sequence or None, default=None</strong><br>
Format (<code>strptime</code>) of the categorical columns that
need to be converted to datetime. If sequence, the n-th format
corresponds to the n-th categorical column that can be successfully
converted. If None, the format is inferred automatically from the
first non NaN value. Values that can not be converted are returned
as NaT.
</p>
<strong>encoding_type: str, default="ordinal"</strong><br>
Type of encoding to use. Choose from:
<ul style="line-height:1.2em;margin-top:5px">
<li>"ordinal": Encode features in increasing order.</li>
<li>"cyclic": Encode features using sine and cosine to capture
their cyclic nature. Note that this creates two columns for
every feature. Non-cyclic features still use ordinal encoding.</li>
</ul>
<p>
<strong>drop_columns: bool, default=True</strong><br>
Whether to drop the original columns after extracting the
features from it.
</p>
<strong>verbose: int, default=0</strong><br>
Verbosity level of the class. Choose from:
<ul style="line-height:1.2em;margin-top:5px">
<li>0 to not print anything.</li>
<li>1 to print basic information.</li>
<li>2 to print detailed information.</li>
</ul>
<strong>logger: str, Logger or None, default=None</strong><br>
<ul style="line-height:1.2em;margin-top:5px">
<li>If None: Doesn't save a logging file.</li>
<li>If str: Name of the log file. Use "auto" for automatic naming.</li>
<li>Else: Python <code>logging.Logger</code> instance.</li>
</ul>
</td>
</tr>
</table>

!!! warning
    Decision trees based algorithms build their split rules according
    to one feature at a time. This means that they will fail to correctly
    process cyclic features since the cos/sin features should be
    considered one single coordinate system.

<br>



## Attributes

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Attributes:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>feature_names_in_: np.array</strong><br>
Names of features seen during fit.
</p>
<p>
<strong>n_features_in_: int</strong><br>
Number of features seen during fit.
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
<td>Transform the data.</td>
</tr>
</table>
<br>


<a name="fit-transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">fit_transform</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/data_cleaning.py#L109">[source]</a>
</span>
</div>
Extract the new features.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like</strong><br>
Feature set with shape=(n_samples, n_features).
</p>
<p>
<strong>y: int, str, sequence or None, default=None</strong><br>
Does nothing. Implemented for continuity of the API.
</p>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>pd.DataFrame</strong><br>
Transformed feature set.
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
<strong>deep: bool, default=True</strong><br>
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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L590">[source]</a>
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
<strong>level: int, default=0</strong><br>
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
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L611">[source]</a>
</span>
</div>
Save the instance to a pickle file.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<strong>filename: str, default="auto"</strong><br>
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
<strong>FeatureExtractor</strong><br>
Estimator instance.
</td>
</tr>
</table>
<br />


<a name="transform"></a>
<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">transform</strong>(X, y=None)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/feature_engineering.py#L117">[source]</a>
</span>
</div>
Extract the new features.
<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>X: dataframe-like</strong><br>
Feature set with shape=(n_samples, n_features).
</p>
<p>
<strong>y: int, str, sequence or None, default=None</strong><br>
Does nothing. Implemented for continuity of the API.
</p>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>pd.DataFrame</strong><br>
Transformed feature set.
</tr>
</table>
<br />



## Example

=== "atom"
    ```python
    from atom import ATOMClassifier
    
    atom = ATOMClassifier(X, y)
    atom.feature_extraction(features=["day", "month"], fmt="%d/%m/%Y")
    ```

=== "stand-alone"
    ```python
    from atom.feature_engineering import FeatureExtractor
    
    feature_extractor = FeatureExtractor(features=["day", "month"], fmt="%d/%m/%Y")
    X = feature_extractor.transform(X)
    ```
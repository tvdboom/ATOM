# ATOMModel
-----------

<div style="font-size:20px">
<em>function</em> atom.api.<strong style="color:#008AB8">ATOMLoader</strong>(estimator,
acronym=None, fullname=None, needs_scaling=False)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L27">[source]</a>
</span>
</div>

Convert an estimator to a model that can be ingested by ATOM. Note that
only estimators that follow [sklearn's API](https://scikit-learn.org/stable/developers/develop.html)
are compatible. Read more about using custom estimators in the [user guide](../../../user_guide/models/#custom-models).

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>estimator: sklearn estimator</strong><br>
Custom model's estimator.
</p>
<p>
<strong>acronym: str or None, optional (default=None)</strong><br>
Model's acronym. Used to call the model from the trainer. If
None, the capital letters in the estimator's __name__ are used
(only if 2 or more, else it uses the entire name).
</p>
<p>
<strong>fullname: str or None, optional (default=None)</strong><br>
Full model's name. If None, the estimator's __name__ is used.
</p>
<p>
<strong>needs_scaling: bool, optional (default=False)</strong><br>
Whether the model needs scaled features. Can not be True for
datasets with more than two dimensions. Read more about this
in the <a href="../../../user_guide/models/#deep-learning">user guide</a>.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>estimator: sklearn estimator</strong><br>
Clone of the provided estimator with custom attributes.
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMRegressor, ATOMModel
from sklearn.linear_model import HuberRegressor

huber =  ATOMModel(HuberRegressor(), name="hub", fullname="Huber")

atom = ATOMRegressor(X, y)
atom.run(huber)
atom.hub.predict(X_new)
```

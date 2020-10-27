# ATOMModel
-----------

<a name="atom"></a>
<pre><em>function</em> <strong style="color:#008AB8">ATOMModel</strong>(estimator, name=None, fullname=None, needs_scaling=True, type="kernel")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L25">[source]</a></div></pre>
<div style="padding-left:3%">
Convert an estimator to a model that can be ingested by ATOM's pipeline.
<br /><br />
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>estimator: class</strong>
<blockquote>
Model's estimator. Can be a class or an instance.
</blockquote>
<strong>name: str, optional (default=None)</strong>
<blockquote>
Model's acronym. Used to call the `model` from the training instance.
 If None, the estimator's name will be used (not recommended).
</blockquote>
<strong>fullname: str, optional (default=None)</strong>
<blockquote>
Full model's name. If None, the estimator's name will be used.
</blockquote>
<strong>needs_scaling: bool, optional (default=True)</strong>
<blockquote>
Whether the model needs scaled features.
</blockquote>
<strong>type: str, optional (default="kernel")</strong>
<blockquote>
Model's type. Choose from:
<ul>
<li>"linear" for linear models.</li>
<li>"tree" for tree-based models.</li>
<li>"kernel" for the remaining models.</li>
</ul>
</blockquote>
</tr>
</table>
</div>
<br />



## Example
----------

```python
from atom import ATOMRegressor, ATOMModel
from sklearn.linear_model import HuberRegressor

model =  ATOMModel(HuberRegressor, name="hub", fullname="Huber", type="linear")

atom = ATOMRegressor(X, y)
atom.run(model)
atom.hub.predict(X_new)
```

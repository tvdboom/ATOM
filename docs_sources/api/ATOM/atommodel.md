# ATOMModel
-----------

<pre><em>function</em> <strong style="color:#008AB8">ATOMModel</strong>(estimator, acronym=None, fullname=None, needs_scaling=False, type="kernel")
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L26">[source]</a></div></pre>
Convert an estimator to a model that can be ingested by ATOM.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>estimator: class</strong>
<blockquote>
Model's estimator. Can be a class or an instance.
</blockquote>
<strong>acronym: str, optional (default=None)</strong>
<blockquote>
Model's acronym. Used to call the model from the trainer.
 If None, the estimator's __name__ will be used (not recommended).
</blockquote>
<strong>fullname: str, optional (default=None)</strong>
<blockquote>
Full model's name. If None, the estimator's __name__ will be used.
</blockquote>
<strong>needs_scaling: bool, optional (default=False)</strong>
<blockquote>
Whether the model needs scaled features. Can not be True for deep learning datasets.
</blockquote>
<strong>type: str, optional (default="kernel")</strong>
<blockquote>
Model's type. Used to select <a href="https://shap.readthedocs.io/en/latest/api.html#core-explainers">shap's explainer</a>
 for plotting. Choose from:
<ul>
<li>"linear" for linear models.</li>
<li>"tree" for tree-based models.</li>
<li>"kernel" for the remaining model types.</li>
</ul>
</blockquote>
</tr>
</table>
<br />



## Example
----------

```python
from atom import ATOMRegressor, ATOMModel
from sklearn.linear_model import HuberRegressor

model =  ATOMModel(HuberRegressor, name="hub", fullname="Huber", needs_scaling=True, type="linear")

atom = ATOMRegressor(X, y)
atom.run(model)
atom.hub.predict(X_new)
```

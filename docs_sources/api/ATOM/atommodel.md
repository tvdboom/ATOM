# ATOMModel
-----------

<pre><em>function</em> <strong style="color:#008AB8">ATOMModel</strong>(estimator, acronym=None, fullname=None, needs_scaling=False)
<div align="right"><a href="https://github.com/tvdboom/ATOM/blob/master/atom/api.py#L25">[source]</a></div></pre>
Convert an estimator to a model that can be ingested by ATOM.
<table width="100%">
<tr>
<td width="15%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="75%" style="background:white;">
<strong>estimator: class</strong>
<blockquote>
Model's estimator. Can be a class or an instance.
</blockquote>
<strong>acronym: str or None, optional (default=None)</strong>
<blockquote>
Model's acronym. Used to call the model from the trainer. If
None, the capital letters in the estimator's __name__ are used
(if 2 or more, else it uses the entire name).
</blockquote>
<strong>fullname: str or None, optional (default=None)</strong>
<blockquote>
Full model's name. If None, the estimator's __name__ is used.
</blockquote>
<strong>needs_scaling: bool, optional (default=False)</strong>
<blockquote>
Whether the model needs scaled features. Can not be True for deep learning datasets.
</blockquote>
</tr>
</table>
<br />



## Example
----------

```python
from atom import ATOMRegressor, ATOMModel
from sklearn.linear_model import HuberRegressor

model =  ATOMModel(HuberRegressor, name="hub", fullname="Huber", needs_scaling=True)

atom = ATOMRegressor(X, y)
atom.run(model)
atom.hub.predict(X_new)
```

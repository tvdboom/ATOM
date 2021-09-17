# plot_ngrams
-------------

<div style="font-size:20px">
<em>method</em> <strong style="color:#008AB8">plot_ngrams</strong>(ngram="words",
index=None, show=10, title=None, figsize=None, filename=None, display=True)
<span style="float:right">
<a href="https://github.com/tvdboom/ATOM/blob/master/atom/plots.py#L3999">[source]</a>
</span>
</div>

Plot n-gram frequencies. The text for the plot is extracted from
the column named `Corpus`. If there is no column with that name,
an exception is raised. If the documents are not tokenized, the
words are separated by spaces.

<table style="font-size:16px">
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Parameters:</strong></td>
<td width="80%" class="td_params">
<p>
<strong>ngram: str or int, optional (default="bigram")</strong><br>
Number of contiguous words to search for (size of n-gram).
Choose from: words (1), bigrams (2), trigrams (3), quadgrams (4).
</p>
<p>
<strong>index: int, tuple, slice or None, optional (default=None)</strong><br>
Indices of the documents in the corpus to include in the
search. If shape (n, m), it selects documents n until m.
If None, it selects all rows in the dataset.
</p>
<p>
<strong>show: int, optional (default=10)</strong><br>
Number of n-grams (ordered by number of occurrences) to show in the plot.
</p>
<p>
<strong>title: str or None, optional (default=None)</strong><br>
Plot's title. If None, the title is left empty.
</p>
<p>
<strong>figsize: tuple, optional (default=None)</strong><br>
Figure's size, format as (x, y). If None, it adapts the
size to the number of n-grams shown.
</p>
<p>
<strong>filename: str or None, optional (default=None)</strong><br>
Name of the file. Use "auto" for automatic naming.
If None, the figure is not saved.
</p>
<p>
<strong>display: bool or None, optional (default=True)</strong><br>
Whether to render the plot. If None, it returns the matplotlib figure.
</p>
</td>
</tr>
<tr>
<td width="20%" class="td_title" style="vertical-align:top"><strong>Returns:</strong></td>
<td width="80%" class="td_params">
<strong>fig: matplotlib.figure.Figure</strong><br>
Plot object. Only returned if <code>display=None</code>.
</td>
</tr>
</table>
<br />



## Example

```python
from atom import ATOMClassifier

atom = ATOMClassifier(X_text, y_text)
atom.textclean()
atom.plot_ngrams("bigrams")
```
<div align="center">
    <img src="../../../img/plots/plot_ngrams.png" alt="plot_ngrams" width="700" height="700"/>
</div>

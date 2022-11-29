# score
-------

:: atom.basemodel:BaseModel.score
    :: signature
    :: head
    :: table:
        - parameters
        - returns


## Example

```pycon
>>> from atom import ATOMClassifier
>>> from sklearn.datasets import load_breast_cancer

>>> # Load data and separate last 5 rows for predictions
>>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
>>> X_new, y_new = X.iloc[-5:], y.iloc[-5:]
>>> X, y = X.iloc[:-5], y.iloc[:-5]

>>> atom = ATOMClassifier(data)
>>> atom.run("LR", metric="f1")

>>> # Using new data
>>> atom.score(X_new, y_new)

1.0

>>> # Using indices
>>> atom.score(slice(10, 92))

0.975609756097561

>>> # Using a custom metric
>>> atom.score(slice(10, 92))

0.9824561403508771  # f1 score

```

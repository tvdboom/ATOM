# decision_function
-------------------

:: atom.basemodel:BaseModel.decision_function
    :: signature
    :: description
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
>>> atom.run("LR")

>>> # Using new data
>>> atom.lr.decision_function(X_new)

0   -20.872124
1   -13.856470
2    -4.496618
3   -23.196171
4    10.066044
Name: decision_function, dtype: float64

>>> # Using indices
>>> atom.lr.decision_function([23, 25])  # Retrieve prediction of rows 23 and 25

23   -15.286529
25    -4.457036
dtype: float64

```

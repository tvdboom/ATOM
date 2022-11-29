# predict_proba
---------------

:: atom.basemodel:BaseModel.predict_proba
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
>>> atom.run("LR")

>>> # Using new data
>>> atom.predict_proba(X_new)

          0             1
0  1.000000  4.036791e-10
1  1.000000  4.856420e-07
2  0.981879  1.812090e-02
3  1.000000  6.081561e-11
4  0.000025  9.999746e-01

>>> # Using indices
>>> atom.predict_proba([23, 25])  # Retrieve prediction of rows 23 and 25

           0         1
23  0.000892  0.999108
25  0.975733  0.024267

```

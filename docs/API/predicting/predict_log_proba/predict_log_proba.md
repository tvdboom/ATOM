# predict_log_proba
-------------------

:: atom.basemodel:BaseModel.predict_log_proba
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
>>> atom.predict_log_proba(X_new)

              0          1
0 -6.024211e-10 -21.230064
1 -3.525172e-07 -14.858167
2 -1.285206e-02  -4.360670
3 -6.837442e-11 -23.406023
4 -1.076932e+01  -0.000021

>>> # Using indices
>>> atom.predict_log_proba([23, 25])  # Retrieve prediction of rows 23 and 25

           0         1
23 -4.191844 -0.015234
25 -5.207398 -0.005491

```

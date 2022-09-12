# predict
---------

:: atom.basemodel:BaseModel.predict
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
>>> atom.predict(X_new)

0    0
1    0
2    0
3    0
4    1
Name: predict, dtype: int32

>>> # Using indices
>>> atom.predict([23, 25])  # Retrieve prediction of rows 23 and 25

23    1
25    1
dtype: int32

```

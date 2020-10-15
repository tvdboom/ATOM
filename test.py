from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from skopt import gp_minimize
from skopt.space.space import Categorical

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

dim = [Categorical([0.8, 0.9, 1.0, 'sqrt', 'log2'], name='max_features')]


def func(x):
    print(x)
    tree = DecisionTreeClassifier(max_features=x[0])
    tree.fit(X_train, y_train)
    return -tree.score(X_test, y_test)


gp_minimize(func, dim, n_calls=6, n_initial_points=2, verbose=True)
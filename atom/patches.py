# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing patches of external modules.

"""


# scikit-optimize ================================================== >>

def inverse_transform(self, Xt):
    """Patch function for inverse_transform method.

    Monkey patch for skopt.space.space.Categorical.inverse_transform
    method to fix bug with string and numerical categories combined.

    Parameters
    ----------
    self: Categorical
        Instance from the patched class.

    Xt: array-like, shape=(n_samples,)
        List of categories.

    Returns
    -------
    Xt: array-like, shape=(n_samples, n_categories)
        The integer categories.

    """
    return self.transformer.inverse_transform(Xt)


def fit(self, X):
    """Patch function for fit method.

    Monkey patch for skopt.space.transformers.LabelEncoder.fit
    method to fix bug with string and numerical categories combined.

    Parameters
    ----------
    self: LabelEncoder
        Instance from the patched class.

    X: array-like, shape=(n_categories,)
        List of categories.

    """
    self.mapping_ = {v: i for i, v in enumerate(X)}
    self.inverse_mapping_ = {i: v for v, i in self.mapping_.items()}
    return self


def transform(self, X):
    """Patch function for skopt.LabelEncoder transform method.

    Monkey patch for skopt.space.transformers.LabelEncoder.transform
    method to fix bug with string and numerical categories combined.

    Parameters
    ----------
    self: LabelEncoder
        Instance from the patched class.

    X: array-like, shape=(n_samples,)
        List of categories.

    Returns
    -------
    Xt: array-like, shape=(n_samples, n_categories)
        The integer categories.

    """
    return [self.mapping_[v] for v in X]


# scikit-learn ===================================================== >>

def score(f):
    """Patch decorator for sklearn's _score function.

    Monkey patch for sklearn.model_selection._validation._score
    function to score pipelines that drop samples during transforming.

    """

    def wrapper(*args, **kwargs):
        args = list(args)  # Convert to list for item assignment
        if len(args[0]) > 1:  # Has transformers
            args[1], args[2] = args[0][:-1].transform(args[1], args[2])

        # Return f(final_estimator, X_transformed, y_transformed, ...)
        return f(args[0][-1], *tuple(args[1:]), **kwargs)

    return wrapper

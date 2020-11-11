# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the Pipeline class.

"""

# Standard packages
import pandas as pd
from inspect import signature
from copy import copy, deepcopy
from typeguard import typechecked

# Own modules
from .basetransformer import BaseTransformer
from .utils import flt


class Pipeline(object):
    """The pipeline class contains the information coupled to a dataset.

    Parameters
    ----------
    T: class
        Parent class from which the pipeline is called.

    name: str
        Name of the pipeline.

    estimators: pd.Series or None, optional (default=None)
        Sequence of estimators fitted on the pipeline's data.

    data: pd.DataFrame or None, optional (default=None)
        Dataset coupled to the pipeline.

    idx: tuple or None, optional (default=None)
        Tuple indicating the train and test sizes.

    mapping: dict or None, optional (default=None)
        Dictionary of the target values mapped to their respective
        encoded integer.

    """

    def __init__(self, T, name, estimators=None, data=None, idx=None, mapping=None):
        # Make copies of the parameters to not overwrite mutable variables
        self.T = T
        self.name = name
        if estimators is None:
            self.estimators = pd.Series([], name=self.name, dtype="object")
        else:
            self.estimators = copy(estimators)
        self.data = deepcopy(data)
        self.idx = copy(idx)
        self.mapping = copy(mapping)

    def __repr__(self):
        repr_ = f"Pipeline: {self.name}"
        for est in self.estimators:
            repr_ += f"\n --> {est.__class__.__name__}"
            for param in signature(est.__init__).parameters:
                if param not in BaseTransformer.attrs + ["self"]:
                    repr_ += f"\n   >>> {param}: {str(flt(getattr(est, param)))}"

        return repr_

    @typechecked
    def rename(self, name: str):
        """Change the pipeline's name."""
        if not name:
            raise ValueError("A pipeline can't have an empty name!")
        else:
            self.name = name
            self.estimators.name = name
            self.T._branches[name] = self.T._branches.pop(self.T._pipe)
            self.T._pipe = name
            self.T.log("Pipeline renamed successfully!")

    def clear(self):
        """Remove the current pipeline."""
        if len(self.T._branches) > 1:
            self.T._branches.pop(self.T._pipe)  # Delete the pipeline
            self.T._pipe = list(self.T._branches.keys())[0]  # Reset the current one
            self.T.log("Pipeline cleared successfully!")
        else:
            raise PermissionError("Can't clear this pipeline!")

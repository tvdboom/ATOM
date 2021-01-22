# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Package's setup code.

"""

import os
import setuptools


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="atom-ml",
    version="v4.3.0",
    license="MIT",
    description="A Python package for fast exploration and experimentation of supervised machine learning pipelines.",
    download_url="https://github.com/tvdboom/ATOM/archive/v4.3.0.tar.gz",
    url="http://github.com/tvdboom/ATOM",
    author="tvdboom",
    author_email="m.524687@gmail.com",
    keywords=["AutoML", "Machine Learning"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["atom"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy >= 1.19.5",
        "scipy >= 1.4.1",
        "pandas >= 1.0.3",
        "tqdm >= 4.35.0",
        "joblib >= 0.16.0",
        "typeguard >= 2.7.1",
        "tabulate >= 0.8.6",
        "scikit-learn >= 0.24",
        "scikit-optimize >= 0.8.1",
        "tpot >= 0.11.7",
        "category-encoders >= 2.1.0",
        "imbalanced-learn >= 0.5.0",
        "pandas-profiling >= 2.3.0",
        "featuretools >= 0.17.0",
        "gplearn >= 0.4.1",
        "matplotlib >= 3.3.0",
        "seaborn >= 0.10.0",
        "shap >= 0.38.1"
    ],
    extras_require={
        "models": [
            "xgboost >= 0.90",
            "lightgbm >= 2.3.0",
            "catboost >= 0.19.1"
        ]
    },
    test_require=[
        "pytest >= 6.1.2",
        "tensorflow >= 2.3.1",
        "keras >= 2.4.3"
    ],
    python_requires=">=3.6"
)

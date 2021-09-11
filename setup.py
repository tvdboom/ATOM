# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: Mavs
Description: Package's setup code.

"""

import setuptools
from atom import __version__


with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements-optional.txt") as f:
    optional_requirements = f.read().splitlines()

with open("requirements-test.txt") as f:
    test_requirements = f.read().splitlines()

setuptools.setup(
    name="atom-ml",
    version=__version__,
    license="MIT",
    description="A Python package for fast exploration of machine learning pipelines",
    download_url=f"https://github.com/tvdboom/ATOM/archive/v{__version__}.tar.gz",
    url="https://github.com/tvdboom/ATOM",
    author="tvdboom",
    author_email="m.524687@gmail.com",
    keywords=["Python package", "Machine Learning", "Modelling", "Data Pipeline"],
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
    install_requires=requirements,
    extras_require={"models": optional_requirements},
    tests_require=test_requirements,
    python_requires=">=3.6"
)

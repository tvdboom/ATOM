# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Package's setup code.

"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='atom-ml',
      version='v4.0.0',
      license='MIT',
      description='A Python AutoML tool for fast exploration and experimentation of supervised machine learning pipelines.',
      download_url='https://github.com/tvdboom/ATOM/archive/v4.0.0.tar.gz',
      url='http://github.com/tvdboom/ATOM',
      author='tvdboom',
      author_email='m.524687@gmail.com',
      keywords=['AutoML', 'Machine Learning'],
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['atom'],
      classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
      ],
      install_requires=[
          'numpy >= 1.17.2',
          'scipy >= 1.4.1',
          'pandas >= 1.0.3',
          'tqdm >= 4.35.0',
          'joblib >= 0.16.0',
          'typeguard >= 2.7.1',
          'tabulate >= 0.8.6',
          'scikit-learn >= 0.23.1',
          'scikit-optimize >= 0.7.4',
          'category-encoders >= 2.1.0',
          'imbalanced-learn >= 0.5.0',
          'pandas-profiling >= 2.3.0',
          'featuretools >= 0.17.0',
          'gplearn >= 0.4.1',
          'matplotlib >= 3.1.0',
          'seaborn >= 0.10.0',
          'xgboost >= 0.90',
          'lightgbm >= 2.3.0',
          'catboost >= 0.19.1'
      ],
      extras_require={
          'xgboost': ['xgboost>=0.90'],
          'lightgbm': ['lightgbm>=2.3.0'],
          'catboost': ['catboost>=0.19.1'],
      },
      python_requires='>=3.6')

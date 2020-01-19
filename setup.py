# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom

'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='atom-ml',
      version='v2.4.0',
      license='MIT',
      description='Package for fast exploration and experimentation of ML tasks',
      download_url='https://github.com/tvdboom/ATOM/archive/v2.4.0.tar.gz',
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
          'numpy>=1.17.2',
          'pandas>=0.25.1',
          'scikit-learn>=0.22',
          'tqdm>=4.35.0',
          'gpyopt>=1.2.5',
          'matplotlib>=3.1.0',
          'seaborn>=0.9.0',
          'imbalanced-learn>=0.5.0',
          'pandas-profiling>=2.3.0',
          'gplearn>=0.4.1',
          'xgboost>=0.90',
          'lightgbm>=2.3.0',
          'catboost>=0.19.1'
      ],
      python_requires='>=3.6')

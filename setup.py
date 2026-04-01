#!/usr/bin/env python

from setuptools import setup

version = dict()
with open('segmetrics/version.py') as f:
    exec(f.read(), version)


setup(
    name='segmetrics',
    version=version['__version__'],
    description='Image segmentation and object detection performance measures for biomedical image analysis and beyond',
    author='Leonid Kostrykin',
    author_email='leonid.kostrykin@bioquant.uni-heidelberg.de',
    url='https://github.com/bmcv/segmetrics',
    license = 'MIT',
    packages = ['segmetrics'],
    test_suite = 'tests.all',
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.20',
        'scikit-image>=0.18,<0.27',
        'scipy',
        'dill',
        'scikit-learn',
        'Deprecated==1.2',
    ],
)

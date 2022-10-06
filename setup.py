#!/usr/bin/env python


from distutils.core import setup

import segmetrics


setup(
    name = 'segmetrics',
    version = segmetrics.VERSION,
    description = 'Segmentation performance measures for biomedical image analysis and beyond',
    author = 'Leonid Kostrykin',
    author_email = 'leonid.kostrykin@bioquant.uni-heidelberg.de',
    url = 'https://kostrykin.com',
    license = 'MIT',
    packages = ['segmetrics'],
)

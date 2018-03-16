#!/usr/bin/env python


from distutils.core import setup

import segmetrics


setup(name='segmetrics',
      version=segmetrics.VERSION,
      description='Metrics for Segmentation Results',
      author='Leonid Kostrykin',
      author_email='leonid.kostrykin@iwr.uni-heidelberg.de',
      url='http://www.bioquant.uni-heidelberg.de/?id=1379',
      packages=['segmetrics'],
)

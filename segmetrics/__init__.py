from segmetrics.boundary import Hausdorff, NSD
from segmetrics.regional import Dice, JaccardCoefficient, JaccardIndex, RandIndex, AdjustedRandIndex, ISBIScore
from segmetrics.detection import FalsePositive, FalseNegative, FalseSplit, FalseMerge
from segmetrics.study import Study

import segmetrics.parallel


VERSION_MAJOR = 1
VERSION_MINOR = 1
VERSION_PATCH = 0

VERSION = '%d.%d%s' % (VERSION_MAJOR, VERSION_MINOR, '.%d' % VERSION_PATCH if VERSION_PATCH > 0 else '')


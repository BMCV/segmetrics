import segmetrics.parallel
from segmetrics.contour import (
    NSD,
    Hausdorff
)
from segmetrics.detection import (
    FalseMerge,
    FalseNegative,
    FalsePositive,
    FalseSplit
)
from segmetrics.regional import (
    AdjustedRandIndex,
    Dice,
    ISBIScore,
    JaccardCoefficient,
    JaccardIndex,
    RandIndex
)
from segmetrics.study import Study

VERSION_MAJOR = 1
VERSION_MINOR = 5
VERSION_PATCH = 0

VERSION = '%d.%d%s' % (
    VERSION_MAJOR,
    VERSION_MINOR,
    '.%d' % VERSION_PATCH if VERSION_PATCH > 0 else '',
)

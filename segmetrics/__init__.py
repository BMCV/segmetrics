from . import parallel
from .contour import (
    NSD,
    Hausdorff,
)
from .detection import (
    FalseMerge,
    FalseNegative,
    FalsePositive,
    FalseSplit,
)
from .regional import (
    AdjustedRandIndex,
    AggregatedJaccardCoefficient,
    Dice,
    ISBIScore,
    JaccardCoefficient,
    JaccardIndex,
    RandIndex,
)
from .study import Study
from .version import __version__

__all__ = [
    '__version__',
    'AdjustedRandIndex',
    'AggregatedJaccardCoefficient',
    'Dice',
    'FalseMerge',
    'FalseNegative',
    'FalsePositive',
    'FalseSplit',
    'Hausdorff',
    'ISBIScore',
    'JaccardCoefficient',
    'JaccardIndex',
    'NSD',
    'RandIndex',
    'Study',
    'VERSION',
    'parallel',
]


#: Alias for backward compatibility.
VERSION = __version__

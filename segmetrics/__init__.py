from . import parallel
from .measures import *
from .measures import __all__ as __all_measures__
from .study import Study
from .version import __version__

__all__ = __all_measures__ + [
    '__version__',
    'Study',
    'VERSION',
    'parallel',
]


#: Alias for backward compatibility.
VERSION = __version__

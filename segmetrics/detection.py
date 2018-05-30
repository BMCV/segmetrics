import numpy as np
from metric import Metric

## Compatibility with Python 3 -->
import sys
if sys.version_info.major == 3: xrange = range
## <-- Compatibility with Python 3


def _assign(assignments, key, value):
    if key not in assignments: assignments[key] = set()
    assignments[key] |= {value}


def _compute_seg_by_ref_assignments(seg, ref, include_background=False):
    seg_by_ref = {}
    if include_background: seg_by_ref[0] = set()
    for seg_label in xrange(1, seg.max() + 1):
        seg_cc = (seg == seg_label)
        if not seg_cc.any(): continue
        ref_label = np.bincount(ref[seg_cc]).argmax()
        _assign(seg_by_ref, ref_label, seg_label)
    return seg_by_ref


def _compute_ref_by_seg_assignments(seg, ref, *args, **kwargs):
    return _compute_seg_by_ref_assignments(ref, seg, *args, **kwargs)


class FalseSplit(Metric):
    """Counts falsely split objects.

    See: Coelho et al., "Nuclear segmentation in microscope cell images: A hand-segmented
    dataset and comparison of algorithms", ISBI 2009
    """

    ACCUMULATIVE = True

    def compute(self, actual):
        seg_by_ref = _compute_seg_by_ref_assignments(actual, self.expected)
        return [sum(len(seg_by_ref[ref_label]) > 1 for ref_label in seg_by_ref.keys() if ref_label > 0)]


class FalseMerge(Metric):
    """Counts falsely merged objects.

    See: Coelho et al., "Nuclear segmentation in microscope cell images: A hand-segmented
    dataset and comparison of algorithms", ISBI 2009
    """

    ACCUMULATIVE = True

    def compute(self, actual):
        ref_by_seg = _compute_ref_by_seg_assignments(actual, self.expected)
        return [sum(len(ref_by_seg[seg_label]) > 1 for seg_label in ref_by_seg.keys() if seg_label > 0)]


class FalsePositive(Metric):
    """Counts falsely detected (added) objects.

    See: Coelho et al., "Nuclear segmentation in microscope cell images: A hand-segmented
    dataset and comparison of algorithms", ISBI 2009
    """

    ACCUMULATIVE = True

    def __init__(self):
        self.result = None

    def compute(self, actual):
        seg_by_ref = _compute_seg_by_ref_assignments(actual, self.expected, include_background=True)
        self.result = np.zeros_like(actual)
        for seg_label in seg_by_ref[0]: self.result[actual == seg_label] = seg_label
        return [len(seg_by_ref[0])]


class FalseNegative(Metric):
    """Counts falsely missed (removed) objects.

    See: Coelho et al., "Nuclear segmentation in microscope cell images: A hand-segmented
    dataset and comparison of algorithms", ISBI 2009
    """

    ACCUMULATIVE = True

    def __init__(self):
        self.result = None

    def compute(self, actual):
        ref_by_seg = _compute_ref_by_seg_assignments(actual, self.expected, include_background=True)
        self.result = np.zeros_like(self.expected)
        for ref_label in ref_by_seg[0]: self.result[self.expected == ref_label] = ref_label
        return [len(ref_by_seg[0])]


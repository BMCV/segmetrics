# -*- coding: utf-8 -*-

import numpy as np
from metric import Metric

## Compatibility with Python 3 -->
import sys
if sys.version_info.major == 3: xrange = range
## <-- Compatibility with Python 3


class Dice(Metric):

    def compute(self, actual):
        ref = self.expected > 0
        res = actual        > 0
        denominator = ref.sum() + res.sum()
        if denominator > 0:
            return [(2. * np.logical_and(ref, res).sum()) / denominator]
        else:
            return [1.]  # result of zero/zero division


class ISBIScore(Metric):
    """Computes segmentation score according to ISBI Cell Tracking Challenge.

    The SEG measure is based on the Jaccard similarity index J = |R ∩ S| / |R ∪ S| of
    the sets of pixels of matching objects R and S, where R denotes the set of pixels
    belonging to a reference object and S denotes the set of pixels belonging to its
    matching segmented object. A ground truth object R and a segmented object S are
    considered matching if and only if |R ∩ S| > 0.5 · |R|. Note that for each
    reference object, there can be at most one segmented object which satisfies the
    detection test. See: http://ctc2015.gryf.fi.muni.cz/Public/Documents/SEG.pdf
    """

    def __init__(self, min_ref_size=1):
        """Instantiates.

        Skips ground truth objects smaller than `min_ref_size` pixels. It is
        recommended to set this value to `2` such that objects of a single pixel in
        size are skipped, but it is set to `1` by default for downwards compatibility.
        """
        assert min_ref_size >= 1, 'min_ref_size must be 1 or larger'
        self.min_ref_size = min_ref_size

    def compute(self, actual):
        results = []
        for ref_label in xrange(1, self.expected.max() + 1):
            ref_cc = (self.expected == ref_label)  # the reference connected component
            ref_cc_size = ref_cc.sum()
            ref_cc_half_size = 0.5 * ref_cc_size
            if ref_cc_size < self.min_ref_size: continue
            actual_cc = None  # the segmented object we compare the reference to
            for actual_candidate_label in set(actual[ref_cc]):
                actual_candidate_cc = (actual == actual_candidate_label)
                overlap = float(np.logical_and(actual_candidate_cc, ref_cc).sum())
                if overlap > ref_cc_half_size:
                    actual_cc = actual_candidate_cc
                    break
            if actual_cc is None:
                jaccard = 0
            else:
                jaccard = overlap / np.logical_or(ref_cc, actual_cc).sum()
            results.append(jaccard)
        return results

import sys
import numpy as np
import sklearn.metrics

from segmetrics.measure import Measure


class Dice(Measure):
    """Defines the Dice coefficient.
    """

    FRACTIONAL = True

    def compute(self, actual):
        ref = self.expected > 0
        res = actual        > 0
        denominator = ref.sum() + res.sum()
        if denominator > 0:
            return [(2. * np.logical_and(ref, res).sum()) / denominator]
        else:
            return [1.]  # result of zero/zero division


class JaccardCoefficient(Measure):
    """Defines the Jaccard coefficient.
    """

    FRACTIONAL = True

    def compute(self, actual):
        ref = self.expected > 0
        res = actual        > 0
        nominator = np.logical_and(ref, res).sum().astype(np.float32)
        denominator = ref.sum() + res.sum() - nominator
        if denominator > 0:
            return [nominator / denominator]
        else:
            return [1.]  # result of zero/zero division


class RandIndex(Measure):
    """Defines the Rand Index.

    See: Coelho et al., "Nuclear segmentation in microscope cell images: A hand-segmented dataset and comparison of algorithms", ISBI 2009
    """

    FRACTIONAL = True

    def compute(self, actual):
        a, b, c, d = self.compute_parts(actual)
        return [(a + d) / float(a + b + c + d)]

    def compute_parts(self, actual):
        R, S = (self.expected > 0), (actual > 0)
        a, b, c, d = 0, 0, 0, 0
        RS = np.empty((2, 2), int)
        RS[0, 0] = ((R == 0) * (S == 0)).sum()
        RS[0, 1] = ((R == 0) * (S == 1)).sum()
        RS[1, 0] = ((R == 1) * (S == 0)).sum()
        RS[1, 1] = ((R == 1) * (S == 1)).sum()
        for rs in np.ndindex(RS.shape):
            n  = RS[rs]
            Ri = rs[0]
            Si = rs[1]
            a += n * (((Ri == R) * (Si == S)).sum() - 1)
            b += n *  ((Ri != R) * (Si == S)).sum()
            c += n *  ((Ri == R) * (Si != S)).sum()
            d += n *  ((Ri != R) * (Si != S)).sum()
        return a, b, c, d


class AdjustedRandIndex(Measure):
    """Defines the Adjusted Rand Index.

    See: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    """

    FRACTIONAL = True

    def compute(self, actual):
        return [sklearn.metrics.adjusted_rand_score(self.expected.flat, actual.flat)]


class JaccardIndex(RandIndex):
    """Defines the Jaccard Index, not to be confused with the Jaccard Similarity Index.

    The Jaccard Index is not upper-bounded. Higher values correspond to better agreement.

    See: Coelho et al., "Nuclear segmentation in microscope cell images: A hand-segmented dataset and comparison of algorithms", ISBI 2009
    """

    FRACTIONAL = False

    def compute(self, actual):
        a, b, c, d = self.compute_parts(actual)
        return [(a + d) / float(b + c + d)]


class ISBIScore(Measure):
    """Defines the SEG performance measure (used in the ISBI Cell Tracking Challenge).

    The SEG measure is based on the Jaccard similarity index :math:`a^2 + b^2 = c^2`
    """

    FRACTIONAL = True

    def __init__(self, min_ref_size=1):
        """Instantiates.

        Skips ground truth objects smaller than ``min_ref_size`` pixels. It is recommended to set this value to ``2`` such that objects of a single pixel in size are skipped, but it is set to ``1`` by default for downwards compatibility.
        """
        assert min_ref_size >= 1, 'min_ref_size must be 1 or larger'
        self.min_ref_size = min_ref_size

    def compute(self, actual):
        results = []
        for ref_label in range(1, self.expected.max() + 1):
            ref_cc = (self.expected == ref_label)  # the reference connected component
            ref_cc_size = ref_cc.sum()
            ref_cc_half_size = 0.5 * ref_cc_size
            if ref_cc_size < self.min_ref_size: continue
            actual_cc = None  # the segmented object we compare the reference to
            for actual_candidate_label in set(actual[ref_cc]) - {0}:
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

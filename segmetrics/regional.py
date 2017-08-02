# -*- coding: utf-8 -*-

import numpy as np
from metric import Metric


class Dice(Metric):

    def compute(self, actual):
        reference = self.expected > 0
        result    = actual        > 0
        return [(2. * np.logical_and(reference, result).sum()) / (reference.sum() + result.sum())]


class Recall(Metric):

    def compute(self, actual):
        tp = np.logical_and(actual  > 0, self.expected > 0).sum()
        fn = np.logical_and(actual == 0, self.expected > 0).sum()
        return [tp / float(tp + fn)]


class Precision(Metric):

    def compute(self, actual):
        tp = np.logical_and(actual > 0, self.expected  > 0).sum()
        fp = np.logical_and(actual > 0, self.expected == 0).sum()
        return [tp / float(tp + fp)]


class Accuracy(Metric):

    def compute(self, actual):
        tp = np.logical_and(actual  > 0, self.expected  > 0).sum()
        tn = np.logical_and(actual == 0, self.expected == 0).sum()
        return [float(tp + tn) / np.prod(self.expected.shape)]


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

    def compute(self, actual):
        results = []
        for ref_label in xrange(1, self.expected.max() + 1):
            ref_cc = (self.expected == ref_label)  # the reference connected component
            ref_cc_half_size = 0.5 * ref_cc.sum()
            if not ref_cc.any(): continue
            actual_cc = None  # the segmented object we compare the reference to
            for actual_candidate_label in xrange(1, actual.max() + 1):
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


# flake8: noqa

import numpy as np
from deprecated import deprecated

from segmetrics.measure import Measure


class COCOmAP(Measure):
    """
    Calculate mean Average Precision for multiple intersection over union thresholds.

    This metric is used in the COCO or 2018 Data Science Bowl challenge.

    .. deprecated:: 1.0
       The implementation of this measure has not undergone any testing or
       development since version 1.0, since the contributor of this measure is
       not involved in segmetrics any longer. This measure will be removed in a
       future version, unless someone finds the time to maintain it.
    """

    @deprecated(version='1.0', reason='This measure will be removed in a future version, unless someone finds the time to maintain it.')
    def __init__(self, min_ref_size=1, iou_thresholds=[0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]):
        """Instantiates.
        
        Parameters
        ----------
        min_ref_size : int
                    Skips ground truth objects smaller than `min_ref_size` pixels.
        iou_thresholds  : list of floats
                    Overlapping thresholds for matching ground and segmented objects.
                    Consideres objects matching if a ground truth object R and a segmented object S
                    satisfy |R ∩ S| / |R ∪ S| > overlap_threshold.
        """
        self.iou_thresholds = iou_thresholds
        self.min_ref_size = min_ref_size

    def _find_match_for_label(self, ref, actual, ref_label, iou_threshold):
        ref_cc = (ref == ref_label)
        for actual_candidate_label in set(actual[ref_cc]) - {0}:
            actual_candidate_cc = (actual == actual_candidate_label)
            overlap = float(np.logical_and(actual_candidate_cc, ref_cc).sum())
            union = np.logical_or(ref_cc, actual_candidate_cc).sum()
            if union == 0:
                continue
            elif overlap/union > iou_threshold:
                return True
        return False

    def compute(self, actual):
        results = []
        for iou_threshold in self.iou_thresholds:
            tp = 0.
            fp = 0.
            fn = 0.

            for actual_label in range(1, actual.max() + 1):
                found_match = self._find_match_for_label(actual, self.expected, actual_label, iou_threshold)
                if found_match:
                    tp += 1
                else:
                    fp += 1

            for ref_label in range(1, self.expected.max() + 1):
                ref_cc = (self.expected == ref_label)  # the reference connected component
                ref_cc_size = ref_cc.sum()
                if ref_cc_size < self.min_ref_size: continue
                found_match = self._find_match_for_label(self.expected, actual, ref_label, iou_threshold)
                if found_match:                    
                    fn += 1

            if tp+fp+fn == 0.:
                results.append(0.)
            else:
                results.append(tp/(tp+fp+fn))
        return [np.mean(results)]

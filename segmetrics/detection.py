import numpy as np
import sys

from segmetrics.measure import Measure


def _assign(assignments, key, value):
    if key not in assignments: assignments[key] = set()
    assignments[key] |= {value}


def _compute_seg_by_ref_assignments(seg, ref, include_background=False):
    seg_by_ref = {}
    if include_background: seg_by_ref[0] = set()
    for seg_label in range(1, seg.max() + 1):
        seg_cc = (seg == seg_label)
        if not seg_cc.any(): continue
        ref_label = np.bincount(ref[seg_cc]).argmax()
        _assign(seg_by_ref, ref_label, seg_label)
    return seg_by_ref


def _compute_ref_by_seg_assignments(seg, ref, *args, **kwargs):
    return _compute_seg_by_ref_assignments(ref, seg, *args, **kwargs)


class FalseSplit(Measure):
    """Counts falsely split objects.

    References:

    - L\. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in microscope cell images: A hand-segmented dataset and comparison of algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('accumulative', False)
        super().__init__(**kwargs)

    def compute(self, actual):
        seg_by_ref = _compute_seg_by_ref_assignments(actual, self.expected)
        return [sum(len(seg_by_ref[ref_label]) > 1 for ref_label in seg_by_ref.keys() if ref_label > 0)]

    def default_name(self):
        return 'Split'


class FalseMerge(Measure):
    """Counts falsely merged objects.

    References:

    - L\. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in microscope cell images: A hand-segmented dataset and comparison of algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('accumulative', False)
        super().__init__(**kwargs)

    def compute(self, actual):
        ref_by_seg = _compute_ref_by_seg_assignments(actual, self.expected)
        return [sum(len(ref_by_seg[seg_label]) > 1 for seg_label in ref_by_seg.keys() if seg_label > 0)]

    def default_name(self):
        return 'Merge'


class FalsePositive(Measure):
    """Counts spurious objects.

    References:

    - L\. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in microscope cell images: A hand-segmented dataset and comparison of algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('accumulative', False)
        super().__init__(**kwargs)
        self.result = None

    def compute(self, actual):
        seg_by_ref = _compute_seg_by_ref_assignments(actual, self.expected, include_background=True)
        self.result = np.zeros_like(actual)
        for seg_label in seg_by_ref[0]: self.result[actual == seg_label] = seg_label
        return [len(seg_by_ref[0])]

    def default_name(self):
        return 'Spurious'


class FalseNegative(Measure):
    """Counts missing objects.

    References:

    - L\. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in microscope cell images: A hand-segmented dataset and comparison of algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('accumulative', False)
        super().__init__(**kwargs)
        self.result = None

    def compute(self, actual):
        ref_by_seg = _compute_ref_by_seg_assignments(actual, self.expected, include_background=True)
        self.result = np.zeros_like(self.expected)
        for ref_label in ref_by_seg[0]: self.result[self.expected == ref_label] = ref_label
        return [len(ref_by_seg[0])]

    def default_name(self):
        return 'Missing'


class COCOmAP(Measure):
    """Calculate mean Average Precision for multiple intersection over union thresholds

    This metric is used in the COCO or 2018 Data Science Bowl challenge.
    """
    
    FRACTIONAL = True

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

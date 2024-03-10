from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
)

import numpy as np

from segmetrics.measure import Measure
from segmetrics.typing import (
    BinaryImage,
    LabelImage,
)


def _assign(
    assignments: Dict,
    key: Any,
    value: Any,
) -> None:
    if key not in assignments:
        assignments[key] = set()
    assignments[key] |= {value}


def _compute_seg_by_ref_assignments(
    seg: LabelImage,
    ref: LabelImage,
    include_background: bool = False,
) -> Dict[int, Set[int]]:
    seg_by_ref: Dict[int, Set[int]] = dict()

    if include_background:
        seg_by_ref[0] = set()

    for seg_label in range(1, seg.max() + 1):
        seg_cc: BinaryImage = (seg == seg_label)

        if not seg_cc.any():
            continue

        ref_label = np.bincount(ref[seg_cc]).argmax()
        _assign(seg_by_ref, ref_label, seg_label)

    return seg_by_ref


def _compute_ref_by_seg_assignments(
    seg: LabelImage,
    ref: LabelImage,
    *args,
    **kwargs,
) -> Dict[int, Set[int]]:
    return _compute_seg_by_ref_assignments(ref, seg, *args, **kwargs)


class FalseSplit(Measure):
    r"""
    Counts falsely split objects.

    References:

    - L\. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in
      microscope cell images: A hand-segmented dataset and comparison of
      algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.
    """

    def compute(self, actual: LabelImage) -> List[float]:
        seg_by_ref = _compute_seg_by_ref_assignments(actual, self.expected)
        return [
            sum(len(seg_by_ref[ref_label]) > 1
                for ref_label in seg_by_ref.keys() if ref_label > 0)
        ]

    def default_name(self) -> str:
        return 'Split'


class FalseMerge(Measure):
    r"""
    Counts falsely merged objects.

    References:

    - L\. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in
      microscope cell images: A hand-segmented dataset and comparison of
      algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.
    """

    def compute(self, actual: LabelImage) -> List[float]:
        ref_by_seg = _compute_ref_by_seg_assignments(actual, self.expected)
        return [
            sum(len(ref_by_seg[seg_label]) > 1
                for seg_label in ref_by_seg.keys() if seg_label > 0)
        ]

    def default_name(self) -> str:
        return 'Merge'


class FalsePositive(Measure):
    r"""
    Counts spurious objects.

    References:

    - L\. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in
      microscope cell images: A hand-segmented dataset and comparison of
      algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.result: Optional[LabelImage] = None

    def compute(self, actual: LabelImage) -> List[float]:
        seg_by_ref = _compute_seg_by_ref_assignments(
            actual,
            self.expected,
            include_background=True,
        )
        self.result = np.zeros_like(actual)
        for seg_label in seg_by_ref[0]:
            self.result[actual == seg_label] = seg_label
        return [len(seg_by_ref[0])]

    def default_name(self) -> str:
        return 'Spurious'


class FalseNegative(Measure):
    r"""
    Counts missing objects.

    References:

    - L\. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in
      microscope cell images: A hand-segmented dataset and comparison of
      algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.result: Optional[LabelImage] = None

    def compute(self, actual: LabelImage) -> List[float]:
        ref_by_seg = _compute_ref_by_seg_assignments(
            actual,
            self.expected,
            include_background=True,
        )
        self.result = np.zeros_like(self.expected)
        for ref_label in ref_by_seg[0]:
            self.result[self.expected == ref_label] = ref_label
        return [len(ref_by_seg[0])]

    def default_name(self) -> str:
        return 'Missing'

import warnings
from typing import (
    List,
    Tuple,
)

import numpy as np
import sklearn.metrics

from segmetrics.measure import (
    AsymmetricMeasureMixin,
    CorrespondanceFunction,
    ImageMeasureMixin,
    Measure,
)
from segmetrics.typing import (
    BinaryImage,
    LabelImage,
)


class RegionalImageMeasure(ImageMeasureMixin, Measure):
    """
    Defines an image-level performance measure which is based on the regions
    of binary volumes (images).
    """

    def __init__(
        self,
        *args,
        correspondance_function: CorrespondanceFunction = 'max',
        **kwargs
    ) -> None:
        super().__init__(
            *args,
            correspondance_function=correspondance_function,
            **kwargs,
        )


class Dice(RegionalImageMeasure):
    r"""
    Defines the Dice coefficient.

    Let :math:`R` be the set of all image pixels corresponding to the ground
    truth segmentation, and :math:`S` the set of those corresponding to the
    segmentation result. Then, the Dice coefficient is defined as

    .. math:: \mathrm{DC} =\frac
        {2 \cdot \left|R \cap S\right|}
        {\left|R\right| + \left|S\right|}

    and attains values between :math:`0` and :math:`1`. Higher values
    correspond to better segmentation performance. Dice corresponds to the
    pixel-based `F1 score`_ (harmonic mean of precision and recall).

    .. _F1 score: https://en.wikipedia.org/wiki/F-score
    """

    def compute(self, actual: LabelImage) -> List[float]:
        ref = self.expected > 0
        res = actual        > 0
        denominator = ref.sum() + res.sum()
        if denominator > 0:
            return [(2. * np.logical_and(ref, res).sum()) / denominator]
        else:
            return [1.]  # result of zero/zero division


class JaccardCoefficient(RegionalImageMeasure):
    r"""
    Defines the Jaccard coefficient.

    Let :math:`R` be the set of all image pixels corresponding to the ground
    truth segmentation, and :math:`S` the set of those corresponding to the
    segmentation result. Then, the Jaccard coefficient is defined as the
    *intersection over the union*,

    .. math:: \mathrm{JC} = \frac
        {\left|R \cap S\right|}
        {\left|R \cup S\right|},

    and attains values between :math:`0` and :math:`1`. Higher values
    correspond to better segmentation performance.

    The Jaccard coefficient equals :math:`\mathrm{JC} = \mathrm{DC} / \left(2 -
    \mathrm{DC}\right)`, where :math:`\mathrm{DC}` is the Dice coefficient.
    Note that this equation only holds for individual :math:`\mathrm{JC}` and
    :math:`\mathrm{DC}` values, but not for sums or mean values thereof.
    """

    def compute(self, actual: LabelImage) -> List[float]:
        ref = self.expected > 0
        res = actual        > 0
        nominator = np.logical_and(ref, res).sum().astype(np.float32)
        denominator = ref.sum() + res.sum() - nominator
        if denominator > 0:
            return [nominator / denominator]
        else:
            return [1.]  # result of zero/zero division

    def default_name(self) -> str:
        return 'Jaccard coef.'


class RandIndex(RegionalImageMeasure):
    r"""
    Defines the Rand index.

    Let :math:`R` be the set of all image pixels corresponding to the ground
    truth segmentation, and :math:`S` the set of those corresponding to the
    segmentation result. Moreover, let :math:`a, b, c, d` be the quantities of
    the events

    .. math::

        &\text{(a) } R_i = R_j \text{ and } S_i = S_j \quad
        &\text{(b) } R_i \neq R_j \text{ and } S_i = S_j \\
        &\text{(c) } R_i = R_j \text{ and } S_i \neq S_j \quad
        &\text{(d) } R_i \neq R_j \text{ and } S_i \neq S_j

    for :math:`i` and :math:`j` ranging over all pair of pixels in :math:`R`
    and :math:`S`. Then, the Rand index is defined as

    .. math:: \mathrm{RI} = \frac{a + d}{a + b + c + d}.

    The Rand index attains values between :math:`0` and :math:`1`. Larger
    values correspond to better segmentation performance. The Rand index
    corresponds to the pixel-based `accuracy score`_.

    .. _accuracy score:
        https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification

    References:

    - L\. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in
      microscope cell images: A hand-segmented dataset and comparison of
      algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.
    """

    def compute(self, actual: LabelImage) -> List[float]:
        a, b, c, d = self.compute_parts(actual)
        return [(a + d) / float(a + b + c + d)]

    def compute_parts(
        self,
        actual: LabelImage,
    ) -> Tuple[float, float, float, float]:
        """
        Computes the values :math:`a`, :math:`b`, :math:`c`, :math:`d`.
        """
        R: BinaryImage
        S: BinaryImage
        R, S = (self.expected > 0), (actual > 0)
        a, b, c, d = 0., 0., 0., 0.
        RS = np.empty((2, 2), np.uint64)
        RS[0, 0] = ((R == 0) * (S == 0)).sum()
        RS[0, 1] = ((R == 0) * (S == 1)).sum()
        RS[1, 0] = ((R == 1) * (S == 0)).sum()
        RS[1, 1] = ((R == 1) * (S == 1)).sum()
        for rs in np.ndindex(RS.shape):
            n : int = RS[rs]
            Ri: int = rs[0]
            Si: int = rs[1]
            a += n * (np.sum((Ri == R) * (Si == S)) - 1)
            b += n *  np.sum((Ri != R) * (Si == S))
            c += n *  np.sum((Ri == R) * (Si != S))
            d += n *  np.sum((Ri != R) * (Si != S))
        return a, b, c, d

    def default_name(self) -> str:
        return 'Rand'


class AdjustedRandIndex(RegionalImageMeasure):
    """
    Defines the adjusted Rand index.

    See:
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    """

    def compute(self, actual: LabelImage) -> List[float]:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            return [
                sklearn.metrics.adjusted_rand_score(
                    self.expected.flat,
                    actual.flat
                )
            ]

    def default_name(self) -> str:
        return 'ARI'


class JaccardIndex(RandIndex):
    r"""
    Defines the Jaccard index, not to be confused with the Jaccard coefficient.

    Let :math:`R` be the set of all image pixels corresponding to the ground
    truth segmentation, and :math:`S` the set of those corresponding to the
    segmentation result. Moreover, let :math:`a, b, c, d` be the quantities of
    the events

    .. math::

        &\text{(a) } R_i = R_j \text{ and } S_i = S_j \quad
        &\text{(b) } R_i \neq R_j \text{ and } S_i = S_j \\
        &\text{(c) } R_i = R_j \text{ and } S_i \neq S_j \quad
        &\text{(d) } R_i \neq R_j \text{ and } S_i \neq S_j

    for :math:`i` and :math:`j` ranging over all pair of pixels in :math:`R`
    and :math:`S`. Then, the Jaccard index is defined as

    .. math:: \mathrm{JI} = \frac{a + d}{b + c + d}.

    The Jaccard index is not upper-bounded. Higher values correspond to better
    segmentation performance.

    References:

    - L\. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in
      microscope cell images: A hand-segmented dataset and comparison of
      algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.
    """

    def compute(self, actual: LabelImage) -> List[float]:
        a, b, c, d = self.compute_parts(actual)
        return [(a + d) / float(b + c + d)]

    def default_name(self) -> str:
        return 'Jaccard index'


class ISBIScore(AsymmetricMeasureMixin, Measure):
    r"""
    Defines the SEG performance measure (used in the ISBI Cell Tracking
    Challenge).

    The SEG measure is based on the Jaccard coefficient :math:`J = \left|R
    \cap S\right| / \left|R \cup S\right|` of the sets of pixels of matching
    objects :math:`R` and :math:`S`, where :math:`R` denotes the set of pixels
    belonging to a reference object and :math:`S` denotes the set of pixels
    belonging to its matching segmented object. A ground truth object
    :math:`R` and a segmented object :math:`S` are considered matching if and
    only if :math:`\left|R \cap S\right| > 0.5 \cdot \left|R\right|`. Note
    that for each reference object, there can be at most one segmented object
    which satisfies the detection test. See:
    http://public.celltrackingchallenge.net/documents/SEG.pdf

    :param min_ref_size:
        Ground truth objects smaller than ``min_ref_size`` pixels are skipped.
        It is reasonable to set this value to ``2`` so that objects of a
        single pixel in size are skipped, since such objects obviously
        correspond to misannotations which distort the performance evaluation.
        However, for compatibility to the official implementation, the value
        is set to ``1`` by default so all ground truth objects are included.

    References:

    - M\. Maska et al., "A benchmark for comparison of cell tracking
      algorithms," Bioinformatics, vol. 30, no. 11, pp. 1609–1617, 2014.
    """

    def __init__(self, min_ref_size: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        assert min_ref_size >= 1, 'min_ref_size must be 1 or larger'
        self.min_ref_size = min_ref_size

    def compute(self, actual: LabelImage) -> List[float]:
        results: List[float] = list()
        for ref_label in range(1, self.expected.max() + 1):

            # The reference connected component
            ref_cc = (self.expected == ref_label)

            ref_cc_size = ref_cc.sum()
            ref_cc_half_size = 0.5 * ref_cc_size

            if ref_cc_size < self.min_ref_size:
                continue

            # The segmented object we compare the reference to
            actual_cc = None

            for actual_candidate_label in set(actual[ref_cc]) - {0}:
                actual_candidate_cc = (actual == actual_candidate_label)
                overlap = float(
                    np.logical_and(actual_candidate_cc, ref_cc).sum()
                )
                if overlap > ref_cc_half_size:
                    actual_cc = actual_candidate_cc
                    break
            if actual_cc is None:
                jaccard = 0
            else:
                jaccard = overlap / np.logical_or(ref_cc, actual_cc).sum()
            results.append(jaccard)
        return results

    def default_name(self) -> str:
        name = 'SEG'
        if self.min_ref_size >= 2:
            name += f' (min_ref_size={self.min_ref_size})'
        return name

import numpy as np
from scipy import ndimage
from skimage import morphology as morph

from segmetrics.measure import ImageMeasure


def _compute_binary_contour(mask, width=1):
    dilation = morph.binary_dilation(mask, morph.disk(width))
    return np.logical_and(dilation, np.logical_not(mask))


def _compute_contour_distance_map(mask):
    contour = _compute_binary_contour(mask)
    return ndimage.distance_transform_edt(np.logical_not(contour))


def _quantile_max(quantile, values):
    if quantile == 1:
        return np.max(values)
    else:
        values = np.sort(values)
        return values[int(quantile * (len(values) - 1))]


class ContourMeasure(ImageMeasure):
    """
    Defines a performance measure which is based on the spatial distances of
    binary volumes (images).
    """

    def __init__(self, correspondance_function='min'):
        super().__init__(correspondance_function)


class Hausdorff(ContourMeasure):
    r"""
    Defines the Hausdorff distsance between two binary images.

    The Hausdorff distsance is not upper-bounded. Lower values correspond to
    better segmentation performance.

    :param mode:
        Specifies how the Hausdorff distance is to be computed.

    :param quantile:
        Specifies the quantile of the Hausdorff distsance. The default
        ``quantile=1`` corresponds to the Hausdorff distance described by
        Bamford (2003). Any other positive value for ``quantile`` corresponds
        to the quantile method introduced by Rucklidge (1997).

    The following values are allowed for the ``mode`` parameter:

    - ``a2e``: Maximum distance of actual foreground to expected foreground.
    - ``e2a``: Maximum distance of expected foreground to actual foreground.
    - ``sym``: Maximum of the two (equivalent to ``symmetric``).

    References:

    - P\. Bamford, "Empirical comparison of cell segmentation algorithms using
      an annotated dataset," in Proc. Int. Conf. Image Proc., 1612 vol. 2,
      2003, pp. II-1073–1076.
    - W\. J. Rucklidge, "Efficiently locating objects using the Hausdorff
      distance." International Journal of computer vision 24.3 (1997): 251-270.
    """

    def __init__(self, mode='sym', quantile=1, **kwargs):
        super().__init__(**kwargs)
        assert mode in ('a2e', 'e2a', 'symmetric', 'sym')
        assert 0 < quantile <= 1

        if mode == 'symmetric':
            mode = 'sym'

        self.mode = mode
        self.quantile = quantile

    def set_expected(self, expected):
        self.expected_contour = _compute_binary_contour(expected > 0)
        self.expected_contour_distance_map = ndimage.distance_transform_edt(
            np.logical_not(self.expected_contour)
        )

    def compute(self, actual):
        actual_contour = _compute_binary_contour(actual > 0)
        if not self.expected_contour.any() or not actual_contour.any():
            return []

        results = []

        if self.mode in ('a2e', 'sym'):
            results.append(self._quantile_max(
                self.expected_contour_distance_map[actual_contour]
            ))

        if self.mode in ('e2a', 'sym'):
            actual_contour_distance_map = ndimage.distance_transform_edt(
                np.logical_not(actual_contour)
            )
            results.append(self._quantile_max(
                actual_contour_distance_map[self.expected_contour]
            ))

        return [max(results)]

    def default_name(self):
        if self.quantile == 1:
            return f'HSD ({self.mode})'
        else:
            return f'HSD ({self.mode}, Q={self.quantile:g})'

    def _quantile_max(self, values):
        return _quantile_max(self.quantile, values)


class NSD(ContourMeasure):
    r"""
    Defines the normalized sum of distsances between two binary images.

    Let :math:`R` be the set of all image pixels corresponding to the ground
    truth segmentation, and :math:`S` the set of those corresponding to the
    segmentation result. Moreover, let :math:`\operatorname{dist}_{\partial R}
    \left(x\right) = \min_{x' \in \partial R}\left\|x - x'\right\|` be the
    Euclidean distance of an image point :math:`x` to the contour
    :math:`\partial R` of the ground truth segmentation :math:`R`. Then, the
    normalized sum of distances is defined as

    .. math::

        \mathrm{NSD} = \sum_{x \in R \triangle S}
        \operatorname{dist}_{\partial R}\left(x\right) / \sum_{x \in R \cup S}
        \operatorname{dist}_{\partial R}\left(x\right),

    where :math:`R \triangle S = \left(R \setminus S\right) \cup \left(S
    \setminus R\right)` is the symmetric difference of :math:`R` and
    :math:`S`. :math:`\mathrm{NSD}` attains values between :math:`0` and
    :math:`1`. Lower values correspond to better segmentation performance.
    """

    def set_expected(self, expected):
        self.expected = (expected > 0)
        self.expected_contour = _compute_binary_contour(self.expected)
        self.expected_contour_distance_map = ndimage.distance_transform_edt(
            np.logical_not(self.expected_contour)
        )

    def compute(self, actual):
        actual = (actual > 0)
        union         = np.logical_or(self.expected, actual)
        intersection  = np.logical_and(self.expected, actual)
        denominator   = self.expected_contour_distance_map[union].sum()
        nominator     = self.expected_contour_distance_map[
            np.logical_and(union, np.logical_not(intersection))
        ].sum()
        return [nominator / (0. + denominator)]

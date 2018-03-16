import numpy as np
import warnings
from scipy import ndimage
from skimage import morphology as morph
from metric import Metric

## Compatibility with Python 3 -->
import sys
if sys.version_info.major == 3: xrange = range
## <-- Compatibility with Python 3


def compute_binary_boundary(mask, width=1):
    dilation = morph.binary_dilation(mask, morph.disk(width))
    return np.logical_and(dilation, np.logical_not(mask))


def compute_boundary_distance_map(mask):
    boundary = compute_binary_boundary(mask)
    return ndimage.morphology.distance_transform_edt(np.logical_not(boundary))


class Hausdorff(Metric):

    def __init__(self, mode='symmetric'):
        """Initializes Hausdorff metric.

        The parameter `mode` specifies how the Hausdorff distance is to be computed:

            a2e        --  maximum distance of actual foreground to expected foreground
            e2a        --  maximum distance of expected foreground to actual foreground
            symmetric  --  maximum of the two
        """
        assert mode in ('a2e', 'e2a', 'symmetric')
        self.mode = mode

    def set_expected(self, expected):
        self.expected_boundary = compute_binary_boundary(expected > 0)
        self.expected_boundary_distance_map = ndimage.morphology.distance_transform_edt(np.logical_not(self.expected_boundary))

    def compute(self, actual):
        actual_boundary = compute_binary_boundary(actual > 0)
        actual_boundary_distance_map = ndimage.morphology.distance_transform_edt(np.logical_not(actual_boundary))
        if not self.expected_boundary.any() or not actual_boundary.any(): return []
        results = []
        if self.mode in ('a2e', 'symmetric'): results.append(self.expected_boundary_distance_map[actual_boundary].max())
        if self.mode in ('e2a', 'symmetric'): results.append(actual_boundary_distance_map[self.expected_boundary].max())
        return [max(results)]


class NSD(Metric):

    def set_expected(self, expected):
        self.expected = (expected > 0)
        self.expected_boundary = compute_binary_boundary(self.expected)
        self.expected_boundary_distance_map = ndimage.morphology.distance_transform_edt(np.logical_not(self.expected_boundary))

    def compute(self, actual):
        actual = (actual > 0)
        actual_boundary = compute_binary_boundary(actual)
        union           = np.logical_or(self.expected, actual)
        intersection    = np.logical_and(self.expected, actual)
        denominator     = self.expected_boundary_distance_map[union].sum()
        nominator       = self.expected_boundary_distance_map[np.logical_and(union, np.logical_not(intersection))].sum()
        return [nominator / (0. + denominator)]


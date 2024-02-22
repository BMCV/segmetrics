import numpy as np
import warnings
import sys

from scipy import ndimage
from skimage import morphology as morph

from segmetrics.measure import Measure
from segmetrics._aux import bbox


def _compute_binary_boundary(mask, width=1):
    dilation = morph.binary_dilation(mask, morph.disk(width))
    return np.logical_and(dilation, np.logical_not(mask))


def _compute_boundary_distance_map(mask):
    boundary = _compute_binary_boundary(mask)
    return ndimage.distance_transform_edt(np.logical_not(boundary))


def _quantile_max(quantile, values):
    if quantile == 1: return np.max(values)
    else:
        values = np.sort(values)
        return values[int(quantile * (len(values) - 1))]


class DistanceMeasure(Measure):
    """Defines a performance measure which is based on the spatial distances of binary volumes (images).
    
    The computation of such measures only regards the union of the individual objects, not the individual objects themselves.
    """

    def object_based(self, *args, **kwargs):
        """Returns measure for computation regarding individual objects (rather than their union).
       
        Positional and keyword arguments are passed through to :class:`ObjectBasedDistanceMeasure`.

        :returns: This measure decorated using :class:`ObjectBasedDistanceMeasure`.
        """
        return ObjectBasedDistanceMeasure(self, *args, **kwargs)


class Hausdorff(DistanceMeasure):
    r"""Defines the Hausdorff distsance between two binary images.
    
    The Hausdorff distsance is not upper-bounded. Lower values correspond to better segmentation performance.
    
    :param mode: Specifies how the Hausdorff distance is to be computed.
    :param quantile: Specifies the quantile of the Hausdorff distsance. The default ``quantile=1`` corresponds to the Hausdorff distance described by Bamford (2003). Any other positive value for ``quantile`` corresponds to the quantile method introduced by Rucklidge (1997).
    
    The following values are allowed for the ``mode`` parameter:

    - ``a2e``: Maximum distance of actual foreground to expected foreground.
    - ``e2a``: Maximum distance of expected foreground to actual foreground.
    - ``sym``: Maximum of the two (equivalent to ``symmetric``).
    
    References:

    - P\. Bamford, "Empirical comparison of cell segmentation algorithms using an annotated dataset," in Proc. Int. Conf. Image Proc., 1612 vol. 2, 2003, pp. II-1073–1076.
    - W\. J. Rucklidge, "Efficiently locating objects using the Hausdorff distance." International Journal of computer vision 24.3 (1997): 251-270.
    """

    def __init__(self, mode='sym', quantile=1, **kwargs):
        super().__init__(**kwargs)
        assert mode in ('a2e', 'e2a', 'symmetric', 'sym')
        assert 0 < quantile <= 1
        if mode == 'symmetric': mode = 'sym'
        self.mode = mode
        self.quantile = quantile

    def set_expected(self, expected):
        self.expected_boundary = _compute_binary_boundary(expected > 0)
        self.expected_boundary_distance_map = ndimage.distance_transform_edt(np.logical_not(self.expected_boundary))

    def compute(self, actual):
        actual_boundary = _compute_binary_boundary(actual > 0)
        if not self.expected_boundary.any() or not actual_boundary.any(): return []
        results = []
        if self.mode in ('a2e', 'sym'): results.append(self._quantile_max(self.expected_boundary_distance_map[actual_boundary]))
        if self.mode in ('e2a', 'sym'):
            actual_boundary_distance_map = ndimage.distance_transform_edt(np.logical_not(actual_boundary))
            results.append(self._quantile_max(actual_boundary_distance_map[self.expected_boundary]))
        return [max(results)]

    def default_name(self):
        if self.quantile == 1:
            return f'HSD ({self.mode})'
        else:
            return f'HSD ({self.mode}, Q={self.quantile:g})'

    def _quantile_max(self, values):
        return _quantile_max(self.quantile, values)


class NSD(DistanceMeasure):
    r"""Defines the normalized sum of distsances between two binary images.
    
    Let :math:`R` be the set of all image pixels corresponding to the ground truth segmentation, and :math:`S` the set of those corresponding to the segmentation result. Moreover, let :math:`\operatorname{dist}_{\partial R}\left(x\right) = \min_{x' \in \partial R}\left\|x - x'\right\|` be the Euclidean distance of an image point :math:`x` to the contour :math:`\partial R` of the ground truth segmentation :math:`R`. Then, the normalized sum of distances is defined as
    
    .. math:: \mathrm{NSD} = \sum_{x \in R \triangle S} \operatorname{dist}_{\partial R}\left(x\right) / \sum_{x \in R \cup S} \operatorname{dist}_{\partial R}\left(x\right),
    
    where :math:`R \triangle S = \left(R \setminus S\right) \cup \left(S \setminus R\right)` is the symmetric difference of :math:`R` and :math:`S`. :math:`\mathrm{NSD}` attains values between :math:`0` and :math:`1`. Lower values correspond to better segmentation performance.
    """

    def set_expected(self, expected):
        self.expected = (expected > 0)
        self.expected_boundary = _compute_binary_boundary(self.expected)
        self.expected_boundary_distance_map = ndimage.distance_transform_edt(np.logical_not(self.expected_boundary))

    def compute(self, actual):
        actual = (actual > 0)
        actual_boundary = _compute_binary_boundary(actual)
        union           = np.logical_or(self.expected, actual)
        intersection    = np.logical_and(self.expected, actual)
        denominator     = self.expected_boundary_distance_map[union].sum()
        nominator       = self.expected_boundary_distance_map[np.logical_and(union, np.logical_not(intersection))].sum()
        return [nominator / (0. + denominator)]


class ObjectBasedDistanceMeasure(Measure):
    """Decorator to apply image-level distance measures on a per-object level.

    Computes the decorated distance measure on a per-object level. Object correspondances between the segmented and the ground truth objects are established on a many-to-many basis, so that the resulting distances are minimal.
    
    :param distance: The image-level distance measure, which is to be decorated.
    :param skip_fn: Specifies whether false-negative detections shall be skipped.
    """
    
    _obj_mapping = (None, None) ## cache

    def __init__(self, distance, skip_fn=False):
        super().__init__()
        self.distance     = distance
        self.skip_fn      = skip_fn
        self.aggregation  = distance.aggregation
        self.nodetections = -1
        
    def set_expected(self, *args, **kwargs):
        super().set_expected(*args, **kwargs)
        ObjectBasedDistanceMeasure._obj_mapping = (None, dict())

    def compute(self, actual):
        results = []
        seg_labels = frozenset(actual.reshape(-1)) - {0}
        
        # Reset the cached object mapping:
        if ObjectBasedDistanceMeasure._obj_mapping[0] is not actual: ObjectBasedDistanceMeasure._obj_mapping = (actual, dict())
            
        for ref_label in set(self.expected.flatten()) - {0}:
            ref_cc = (self.expected == ref_label)

            # If there were no detections at all, then no distances can be determined:
            if len(seg_labels) == 0:
                if self.nodetections >= 0:
                    results.append(self.nodetections)
                continue
            
            if self.skip_fn:
                potentially_closest_seg_labels = frozenset(actual[ref_cc].reshape(-1)) - {0}
            else:

                # Query the cached object mapping:
                if ref_label in self._obj_mapping[1]: ## cache hit

                    potentially_closest_seg_labels = ObjectBasedDistanceMeasure._obj_mapping[1][ref_label]

                else: ## cache miss

                    # First, we determine the set of potentially "closest" segmented objects:
                    ref_distancemap = ndimage.distance_transform_edt(~ref_cc)
                    closest_potential_seg_label = min(seg_labels, key=lambda seg_label: ref_distancemap[actual == seg_label].min())
                    max_potential_seg_label_distance = ref_distancemap[actual == closest_potential_seg_label].max()
                    potentially_closest_seg_labels = [seg_label for seg_label in seg_labels if ref_distancemap[actual == seg_label].min() <= max_potential_seg_label_distance]
                    ObjectBasedDistanceMeasure._obj_mapping[1][ref_label] = potentially_closest_seg_labels

            # If not a single object was detected, the distance is undefined:
            if len(potentially_closest_seg_labels) == 0: continue
            
            distances = []
            for seg_label in potentially_closest_seg_labels:
                seg_cc = (actual == seg_label)
                _bbox  = bbox(ref_cc, seg_cc, margin=1)[0]
                self.distance.set_expected(ref_cc[_bbox].astype('uint8'))
                distance = self.distance.compute(seg_cc[_bbox].astype('uint8'))
                assert len(distance) == 1
                distances.append(distance[0])
            results.append(min(distances))
        return results

    def default_name(self):
        name = f'Ob. {self.distance.default_name()}'
        if self.skip_fn:
            skip_fn_hint = f'skip_fn={self.skip_fn}'
            if name.endswith(')'):
                name = name[:-1] + f', {skip_fn_hint})'
            else:
                name += f' ({skip_fn_hint})'
        return name


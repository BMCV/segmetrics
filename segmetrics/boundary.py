import numpy as np
import warnings
import sys

from scipy import ndimage
from skimage import morphology as morph

from segmetrics.measure import Measure
from segmetrics._aux import bbox


def compute_binary_boundary(mask, width=1):
    dilation = morph.binary_dilation(mask, morph.disk(width))
    return np.logical_and(dilation, np.logical_not(mask))


def compute_boundary_distance_map(mask):
    boundary = compute_binary_boundary(mask)
    return ndimage.morphology.distance_transform_edt(np.logical_not(boundary))


class DistanceMeasure(Measure):

    def object_based(self):
        return ObjectBasedDistanceMeasure(self)


class Hausdorff(DistanceMeasure):

    def __init__(self, mode='symmetric'):
        """Initializes Hausdorff metric.

        The parameter `mode` specifies how the Hausdorff distance is to be computed:

            a2e        --  maximum distance of actual foreground to expected foreground
            e2a        --  maximum distance of expected foreground to actual foreground
            symmetric  --  maximum of the two

        Passing the value `sym` is equivalent to `symmetric`.
        """
        assert mode in ('a2e', 'e2a', 'symmetric', 'sym')
        if mode == 'symmetric': mode = 'sym'
        self.mode = mode

    def set_expected(self, expected):
        self.expected_boundary = compute_binary_boundary(expected > 0)
        self.expected_boundary_distance_map = ndimage.morphology.distance_transform_edt(np.logical_not(self.expected_boundary))

    def compute(self, actual):
        actual_boundary = compute_binary_boundary(actual > 0)
        if not self.expected_boundary.any() or not actual_boundary.any(): return []
        results = []
        if self.mode in ('a2e', 'sym'): results.append(self.expected_boundary_distance_map[actual_boundary].max())
        if self.mode in ('e2a', 'sym'):
            actual_boundary_distance_map = ndimage.morphology.distance_transform_edt(np.logical_not(actual_boundary))
            results.append(actual_boundary_distance_map[self.expected_boundary].max())
        return [max(results)]


class NSD(DistanceMeasure):

    FRACTIONAL = True

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


class ObjectBasedDistanceMeasure(Measure):
    """Decorator to apply image-level distance measures on a per-object level.

    Computes the decorated distance measure on a per-object level. Correspondances
    between the segmented and the ground truth objects are established on a n-to-m
    basis, such that the resulting distances are minimal.
    """
    
    obj_mapping = (None, None) ## cache

    def __init__(self, distance, skip_fn=False):
        """Instantiates.

        Parameters
        ----------
        distance : Measure
                   The image-level distance measure, which is to be decorated.
        skip_fn  : bool
                   Specifies whether false-negative detections shall be skipped.
        """
        self.distance     = distance
        self.skip_fn      = skip_fn
        self.FRACTIONAL   = distance.FRACTIONAL
        self.ACCUMULATIVE = distance.ACCUMULATIVE
        self.nodetections = -1
        
    def set_expected(self, *args, **kwargs):
        super().set_expected(*args, **kwargs)
        ObjectBasedDistanceMeasure.obj_mapping = (None, dict())

    def compute(self, actual):
        results = []
        seg_labels = frozenset(actual.reshape(-1)) - {0}
        
        # Reset the cached object mapping:
        if ObjectBasedDistanceMeasure.obj_mapping[0] is not actual: ObjectBasedDistanceMeasure.obj_mapping = (actual, dict())
            
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
                if ref_label in self.obj_mapping[1]: ## cache hit

                    potentially_closest_seg_labels = ObjectBasedDistanceMeasure.obj_mapping[1][ref_label]

                else: ## cache miss

                    # First, we determine the set of potentially "closest" segmented objects:
                    ref_distancemap = ndimage.distance_transform_edt(~ref_cc)
                    closest_potential_seg_label = min(seg_labels, key=lambda seg_label: ref_distancemap[actual == seg_label].min())
                    max_potential_seg_label_distance = ref_distancemap[actual == closest_potential_seg_label].max()
                    potentially_closest_seg_labels = [seg_label for seg_label in seg_labels if ref_distancemap[actual == seg_label].min() <= max_potential_seg_label_distance]
                    ObjectBasedDistanceMeasure.obj_mapping[1][ref_label] = potentially_closest_seg_labels

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

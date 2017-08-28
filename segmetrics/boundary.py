import numpy as np
import warnings
from scipy import ndimage
from skimage import morphology as morph
from metric import Metric


class Hausdorff(Metric):

    def set_expected(self, expected):
        self.expected = Hausdorff.prepare_groundtruth(expected)

    def compute(self, actual):
        results = []
        for i in xrange(1, actual.max() + 1):
            cc = (actual == i)
            if not cc.any(): continue
            cc_boundary = Hausdorff.binary_boundary(cc)
            result = Hausdorff.compute_from_contour(cc_boundary, self.expected)
            if result is not None: results.append(result)
            else: warnings.warn('cannot compute Hausdorff distance since no reference object is available')
        return results

    @staticmethod
    def prepare_groundtruth(expected):
        groundtruth = {}
        for i in xrange(1, expected.max() + 1):
            cc = (expected == i)
            if not cc.any(): continue
            cc_centroid = ndimage.measurements.center_of_mass(cc)
            cc_boundary = Hausdorff.binary_boundary(cc)
            cc_distance = ndimage.morphology.distance_transform_edt(np.logical_not(cc_boundary))
            groundtruth[cc_centroid] = cc_distance
        return groundtruth

    @staticmethod
    def compute_from_contour(segmented_contour, groundtruth):
        segmented_centroid = ndimage.measurements.center_of_mass(segmented_contour)
        nearest_centroid, nearest_distance = None, float('inf')
        for centroid in groundtruth:
            distance = np.linalg.norm(np.subtract(centroid, segmented_centroid))
            if distance < nearest_distance:
                nearest_centroid = centroid
                nearest_distance = distance
        if nearest_centroid is None: return None
        segmented_contour_distances = groundtruth[nearest_centroid][segmented_contour.astype(bool)]
        return segmented_contour_distances.max()

    @staticmethod
    def binary_boundary(mask, width=1):
        dilation = morph.binary_dilation(mask, morph.disk(width))
        return np.logical_and(dilation, np.logical_not(mask))


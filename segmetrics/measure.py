from scipy import ndimage

from segmetrics._aux import bbox


class Measure:
    """
    Defines a performance measure.

    :param aggregation:
        Indicates whether the results of this performance measure are
        aggregated by summation (``sum``), by averaging (``mean``), or by
        computing the proportion with respect to the number of annotated
        objects (``obj-mean``).
    """

    def __init__(self, aggregation='mean'):
        self.aggregation = aggregation

    def set_expected(self, expected):
        """
        Sets the expected result for evaluation.

        :param expected:
            An image containing uniquely labeled object masks corresponding to
            the ground truth.
        """
        self.expected = expected

    def compute(self, actual):
        """
        Computes the performance measure for the given segmentation results
        based on the previously set expected result.

        :param actual:
            An image containing uniquely labeled object masks corresponding to
            the segmentation results.
        """
        return NotImplemented

    def default_name(self):
        """
        Returns the default name of this measure.
        """
        return type(self).__name__


class ImageMeasureMixin:
    """
    Defines an image-level performance measure.

    The computation of such measures only regards the union of the individual
    objects, not the individual objects themselves.

    :param correspondance_function:
        Determines how the object correspondances between the segmented and
        the ground truth objects are determined when using the
        :meth:`object_based` method. The correspondances are established by
        choosing the segmented object for each ground truth object, for which
        the obtained scores are either minimal (``min``) or maximal (``max``).
    """

    def __init__(self, *args, correspondance_function, **kwargs):
        super().__init__(*args, **kwargs)
        assert correspondance_function in (
            'min',
            'max',
        )
        self.correspondance_function = correspondance_function

    def object_based(self, *args, **kwargs):
        """
        Returns measure for comparison regarding the individual objects (as
        opposed to only considering their union).

        Positional and keyword arguments are passed through to
        :class:`ObjectMeasureAdapter`.

        :returns:
            This measure decorated by :class:`ObjectMeasureAdapter`.
        """
        return ObjectMeasureAdapter(
            self,
            *args,
            correspondance_function={
                'min': min,
                'max': max,
            }[self.correspondance_function],
            **kwargs
        )


class AsymmetricMeasureMixin:

    def reversed(self, *args, **kwargs):
        return ReverseMeasureAdapter(self, *args, **kwargs)

    def symmetric(self, *args, **kwargs):
        return SymmetricMeasureAdapter(self, self.reversed(), *args, **kwargs)


class ObjectMeasureAdapter(AsymmetricMeasureMixin, Measure):
    """
    Adapter to use image-level measures on a per-object level.

    Computes the underlying image-level measure on a per-object level. Object
    correspondances between the segmented and the ground truth objects are
    established by choosing the segmented object for each ground truth object,
    for which the obtained scores are either minimal or maximal.

    :param measure:
        The underlying image-level measure.

    :param correspondance_function:
        Determines the object correspondances by reducing a sequence of scores
        to a single score value.
    """

    _obj_mapping = (None, None)  # cache

    def __init__(self, measure, correspondance_function):
        super().__init__()
        self.measure      = measure
        self.aggregation  = measure.aggregation
        self.nodetections = -1  # value to be used if detections are empty
        self.correspondance_function = correspondance_function

    def set_expected(self, *args, **kwargs):
        super().set_expected(*args, **kwargs)
        ObjectMeasureAdapter._obj_mapping = (None, dict())

    def compute(self, actual):
        results = []
        seg_labels = frozenset(actual.reshape(-1)) - {0}

        # Reset the cached object mapping:
        if ObjectMeasureAdapter._obj_mapping[0] is not actual:
            ObjectMeasureAdapter._obj_mapping = (actual, dict())

        for ref_label in set(self.expected.flatten()) - {0}:
            ref_cc = (self.expected == ref_label)

            # If there were no detections, then there are no correspondances,
            # and thus no object-level scores can be determined:
            if len(seg_labels) == 0:
                if self.nodetections >= 0:
                    results.append(self.nodetections)
                continue

            # Query the cached object correspondance candidates:
            if ref_label in self._obj_mapping[1]:  # cache hit

                correspondance_candidates = \
                    ObjectMeasureAdapter._obj_mapping[1][ref_label]

            else:  # cache miss

                # We restrict the search for potentially corresponding objects
                # to a meaningful region. To do so, we first determine the
                # distance within which potentially corresponding objects will
                # be considered. This is the distance to the furthest point of
                # the closest object:
                ref_distancemap = ndimage.distance_transform_edt(~ref_cc)
                closest_seg_label = min(
                    seg_labels,
                    key=lambda seg_label: ref_distancemap[
                            actual == seg_label
                        ].min(),
                )
                max_correspondance_candidates_distance = ref_distancemap[
                    actual == closest_seg_label
                ].max()

                # Second, narrow the set of potentially corresponding objects
                # by finding the labels of objects within the maximum distance:
                correspondance_candidates = [
                    seg_label for seg_label in seg_labels
                    if ref_distancemap[
                        actual == seg_label
                    ].min() <= max_correspondance_candidates_distance
                ]
                ObjectMeasureAdapter._obj_mapping[1][
                    ref_label
                ] = correspondance_candidates

            scores = list()
            for seg_label in correspondance_candidates:
                seg_cc = (actual == seg_label)
                _bbox  = bbox(ref_cc, seg_cc, margin=1)[0]
                self.measure.set_expected(ref_cc[_bbox].astype('uint8'))
                score = self.measure.compute(seg_cc[_bbox].astype('uint8'))
                assert len(score) == 1
                scores.append(score[0])
            results.append(self.correspondance_function(scores))
        return results

    def default_name(self):
        return f'Ob. {self.measure.default_name()}'


class ReverseMeasureAdapter(Measure):

    def __init__(self, measure, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measure     = measure
        self.aggregation = measure.aggregation

    def compute(self, actual):
        self.measure.set_expected(actual)
        return self.measure.compute(self.expected)

    def default_name(self):
        return f'Rev. {self.measure.default_name()}'


class SymmetricMeasureAdapter(Measure):

    def __init__(
        self,
        measure1,
        measure2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert measure1.aggregation == measure2.aggregation
        self.measure1    = measure1
        self.measure2    = measure2
        self.aggregation = measure1.aggregation

    def set_expected(self, expected):
        self.measure1.set_expected(expected)
        self.measure2.set_expected(expected)
        super().set_expected(expected)

    def compute(self, actual):
        results1 = self.measure1.compute(actual)
        results2 = self.measure2.compute(actual)
        return results1 + results2

    def default_name(self):
        return f'Sym. {self.measure1.default_name()}'

from typing import (
    Callable,
    List,
    Literal,
    Protocol,
    get_args,
    runtime_checkable,
)

from scipy import ndimage

from segmetrics._aux import bbox
from segmetrics.typing import LabelImage

AggregationType = Literal[
    'sum',
    'mean',
    'geometric-mean',
    'object-mean',
]

CorrespondanceFunction = Literal[
    'min',
    'max',
]


@runtime_checkable
class MeasureProtocol(Protocol):
    """
    Type protocol of performance measures.
    """

    @property
    def aggregation(self) -> AggregationType:
        """
        Indicates whether the results of this performance measure are
        aggregated by summation (``sum``), by averaging (``mean``), by using
        the geometric mean (``geometric-mean``), or by computing the proportion
        with respect to the number of annotated objects (``object-mean``).
        """
        ...

    def set_expected(self, expected: LabelImage) -> None:
        """
        Sets the expected result for evaluation.

        :param expected:
            An image containing uniquely labeled object masks corresponding to
            the ground truth.
        """
        ...

    def compute(self, actual: LabelImage) -> List[float]:
        """
        Computes the performance measure for the given segmentation results
        based on the previously set expected result.

        :param actual:
            An image containing uniquely labeled object masks corresponding to
            the segmentation results.
        """
        ...

    def default_name(self) -> str:
        """
        Returns the default name of this measure.
        """
        ...


class Measure(MeasureProtocol):
    """
    Defines a performance measure.

    :param aggregation:
        Controls whether the results of this performance measure are
        aggregated by summation (``sum``), by averaging (``mean``), by using
        the geometric mean (``geometric-mean``), or by computing the
        proportion with respect to the number of annotated objects
        (``object-mean``).
    """

    def __init__(self, aggregation: AggregationType = 'mean') -> None:
        assert aggregation in get_args(AggregationType)
        self._aggregation: AggregationType = aggregation

    @property
    def aggregation(self) -> AggregationType:
        return self._aggregation

    def set_expected(self, expected: LabelImage) -> None:
        self.expected = expected

    def compute(self, actual: LabelImage) -> List[float]:
        return NotImplemented

    def default_name(self) -> str:
        return type(self).__name__


class ImageMeasureMixin(MeasureProtocol):
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

    def __init__(
        self,
        *args,
        correspondance_function: CorrespondanceFunction,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert correspondance_function in get_args(CorrespondanceFunction)
        self.correspondance_function: CorrespondanceFunction
        self.correspondance_function = correspondance_function

    def object_based(self, **kwargs) -> Measure:
        """
        Returns measure for comparing the individual objects (as opposed to
        only considering the union thereof).

        Keyword arguments are passed through to :class:`ObjectMeasureAdapter`.

        :returns:
            This measure decorated by :class:`ObjectMeasureAdapter`.
        """
        return ObjectMeasureAdapter(
            measure=self,
            correspondance_function={
                'min': min,
                'max': max,
            }[self.correspondance_function],
            **kwargs
        )


class AsymmetricMeasureMixin(MeasureProtocol):
    """
    Defines an asymmetric performance measure.

    Symmetric performance measures are guaranteed to yield the same results
    when the actual and the expected segmentation masks are swapped.
    Asymmetric performance measures can yield different results when the
    results are swapped.
    """

    def reversed(self, **kwargs) -> Measure:
        """
        Returns a measure for comparing the underlying asymmetric performance
        measure measure in the opposite direction (i.e. swapping the actual
        and the expected segmentation masks).

        Keyword arguments are passed through to :class:`ReverseMeasureAdapter`.

        :returns:
            This measure decorated by :class:`ReverseMeasureAdapter`.
        """
        return ReverseMeasureAdapter(self, **kwargs)

    def symmetric(self, **kwargs) -> Measure:
        """
        Returns a bidirectional variant of the underlying asymmetric
        performance measure. The underlying performance measure is used for
        computation of performance values in both direction (i.e. with the
        original segmentation masks, and, in addition, with actual and
        expected segmentation masks swapped).

        Keyword arguments are passed through to
        :class:`SymmetricMeasureAdapter`.

        :returns:
            This measure decorated by :class:`SymmetricMeasureAdapter`.
        """
        return SymmetricMeasureAdapter(self, self.reversed(), **kwargs)


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

    def __init__(
        self,
        measure: MeasureProtocol,
        correspondance_function: Callable[[List[float]], float],
        **kwargs
    ) -> None:
        super().__init__(aggregation=measure.aggregation, **kwargs)
        self.measure      = measure
        self.nodetections = -1  # value to be used if detections are empty
        self.correspondance_function = correspondance_function

    def compute(self, actual: LabelImage) -> List[float]:
        results: List[float] = list()
        seg_labels = frozenset(actual.reshape(-1)) - {0}

        for ref_label in set(self.expected.flatten()) - {0}:
            ref_cc = (self.expected == ref_label)

            # If there were no detections, then there are no correspondances,
            # and thus no object-level scores can be determined:
            if len(seg_labels) == 0:
                if self.nodetections >= 0:
                    results.append(self.nodetections)
                continue

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

            scores: List[float] = list()
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

    def __init__(self, measure: MeasureProtocol, **kwargs) -> None:
        super().__init__(aggregation=measure.aggregation, **kwargs)
        self.measure = measure

    def compute(self, actual: LabelImage) -> List[float]:
        self.measure.set_expected(actual)
        return self.measure.compute(self.expected)

    def default_name(self) -> str:
        return f'Rev. {self.measure.default_name()}'


class SymmetricMeasureAdapter(Measure):

    def __init__(
        self,
        measure1: MeasureProtocol,
        measure2: MeasureProtocol,
        **kwargs,
    ) -> None:
        super().__init__(aggregation=measure1.aggregation, **kwargs)
        assert measure1.aggregation == measure2.aggregation
        self.measure1 = measure1
        self.measure2 = measure2

    def set_expected(self, expected: LabelImage) -> None:
        self.measure1.set_expected(expected)
        self.measure2.set_expected(expected)
        super().set_expected(expected)

    def compute(self, actual: LabelImage) -> List[float]:
        results1 = self.measure1.compute(actual)
        results2 = self.measure2.compute(actual)
        return results1 + results2

    def default_name(self) -> str:
        return f'Sym. {self.measure1.default_name()}'

import skimage.measure


class Study:

    def __init__(self):
        self.measures = {}
        self.results  = {}

    def add_measure(self, measure, name=None):
        if name is None: name = '%d' % id(measure)
        self.measures[name] = measure
        self.results [name] = []

    def set_expected(self, expected, unique=True):
        """Sets the `expected` ground truth image.
        
        The background must be labeled as 0. Negative object labels are
        forbidden. If `unique` is `True`, it is assumed that all objects
        are labeled uniquely. Set it to `False`, if this is not sure
        (e.g., if the ground truth image is binary).
        """
        assert expected.min() == 0, 'mis-labeled ground truth'
        if not unique: expected = label(expected)
        for measure_name in self.measures:
            measure = self.measures[measure_name]
            measure.set_expected(expected)

    def process(self, actual):
        intermediate_results = {}
        for measure_name in self.measures:
            measure = self.measures[measure_name]
            result = measure.compute(actual)
            self.results[measure_name] += result
            intermediate_results[measure_name] = result
        return intermediate_results


def label(im, background=0, neighbors=4):
    return skimage.measure.label(im, background=background, neighbors=neighbors) + 1


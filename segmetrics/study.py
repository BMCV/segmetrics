class Study:

    def __init__(self):
        self.measures = {}
        self.results  = {}

    def add_measure(self, measure, name=None):
        if name is None: name = '%d' % id(measure)
        self.measures[name] = measure
        self.results [name] = []

    def set_expected(self, expected):
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


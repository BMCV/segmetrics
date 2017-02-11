class Study:

    def __init__(self):
        self.measures = {}
        self.results  = {}

    def add_measure(self, measure, name):
        self.measures[name] = measure
        self.results [name] = []

    def process(self, actual, expected):
        intermediate_results = {}
        for measure_name in self.measures:
            measure = self.measures[measure_name]
            result = measure.compute(actual, expected)
            self.results[measure_name] += result
            intermediate_results[measure_name] = result
        return intermediate_results


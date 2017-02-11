import numpy as np


class Dice:

    def compute(self, actual, expected):
        reference = expected > 0
        result    = actual   > 0
        return [(2. * np.logical_and(reference, result).sum()) / (reference.sum() + result.sum())]


class Recall:

    def compute(self, actual, expected):
        tp = np.logical_and(actual  > 0, expected > 0).sum()
        fn = np.logical_and(actual == 0, expected > 0).sum()
        return [tp / float(tp + fn)]


class Precision:

    def compute(self, actual, expected):
        tp = np.logical_and(actual > 0, expected  > 0).sum()
        fp = np.logical_and(actual > 0, expected == 0).sum()
        return [tp / float(tp + fp)]


class Accuracy:

    def compute(self, actual, expected):
        tp = np.logical_and(actual  > 0, expected  > 0).sum()
        tn = np.logical_and(actual == 0, expected == 0).sum()
        return [float(tp + tn) / expected.sum()]


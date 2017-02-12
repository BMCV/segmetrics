import numpy as np
from metric import Metric


class Dice(Metric):

    def compute(self, actual):
        reference = self.expected > 0
        result    = actual        > 0
        return [(2. * np.logical_and(reference, result).sum()) / (reference.sum() + result.sum())]


class Recall(Metric):

    def compute(self, actual):
        tp = np.logical_and(actual  > 0, self.expected > 0).sum()
        fn = np.logical_and(actual == 0, self.expected > 0).sum()
        return [tp / float(tp + fn)]


class Precision(Metric):

    def compute(self, actual):
        tp = np.logical_and(actual > 0, self.expected  > 0).sum()
        fp = np.logical_and(actual > 0, self.expected == 0).sum()
        return [tp / float(tp + fp)]


class Accuracy(Metric):

    def compute(self, actual):
        tp = np.logical_and(actual  > 0, self.expected  > 0).sum()
        tn = np.logical_and(actual == 0, self.expected == 0).sum()
        return [float(tp + tn) / np.prod(self.expected.shape)]


class Measure:
    """Defines a performance measure.

    :param accumulative: Indicates whether the results from this measure are aggregated by summing (``True``) or by taking the average (``False``).
    """

    """Indicates whether the results from this measure should be presented as percentages."""
    FRACTIONAL = False

    def __init__(self, accumulative=False):
        self.accumulative = accumulative

    def set_expected(self, expected):
        """Sets the expected result for evaluation.
        
        :param expected: An image containing uniquely labeled object masks corresponding to the ground truth.
        """
        self.expected = expected
        
    def compute(self, actual):
        """Computes the performance measure for the given segmentation results based on the previously set expected result.
        
        :param actual: An image containing uniquely labeled object masks corresponding to the segmentation results.
        """
        return NotImplemented

    def default_name(self):
        """Returns the default name of this measure.
        """
        return type(self).__name__

class Measure:
    """Defines a performance measure.
    """

    """Indicates whether the results from this measure should be presented as percentages."""
    FRACTIONAL   = False
    
    """Indicates whether the results from this measure are aggregated by summing (``True``) or by taking the average (``False``)."""
    ACCUMULATIVE = False

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

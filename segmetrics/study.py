import skimage.measure
import numpy as np
import math
import itertools
import sys
import csv

from segmetrics.measure import Measure


def _is_boolean(narray):
    return narray.dtype == bool


def _is_integral(narray):
    return issubclass(narray.dtype.type, np.integer)


def _get_labeled(narray, unique, img_hint):
    assert _is_integral(narray) or _is_boolean(narray), 'illegal %s dtype' % img_hint
    assert not unique or not _is_boolean(narray), 'with unique=True a non-boolean %s is expected' % img_hint
    return narray if unique else _label(narray)


def _get_skimage_measure_label_bg_label():
    """Determines the background label generated by the `label` routine of `skimage.measure`.

    The value of the background label varies from version to version.
    """
    return skimage.measure.label(np.array([[1, 0]]), background=0).min()

_SKIMAGE_MEASURE_LABEL_BF_LABEL = _get_skimage_measure_label_bg_label()


def _label(im, background=0, neighbors=4):
    """Labels the given image `im`.
    
    Returns a labeled version of `im` where `background` is labeled with
    0 and all other connected components are labeled with values larger
    than or equal to 1.
    """
    assert neighbors in (4, 8)
    connectivity = int(math.log2(neighbors)) - 1
    return skimage.measure.label(im, background=background, connectivity=connectivity) \
        - _SKIMAGE_MEASURE_LABEL_BF_LABEL # this is 1 in older versions and 0 in newer


def _aggregate(measure, values):
    fnc = np.sum if measure.accumulative else np.mean
    return fnc(values)


class Study:
    """Computes different performance measures for different image data.
    
    Performance measures must be added prior to performing evaluation.
    """

    def __init__(self):
        self.measures   = dict()
        self.sample_ids = list()
        self.results    = dict()
        self.results_cache = dict()
        self.csv_sample_id_column_name = 'Sample'

    def merge(self, other, sample_ids='all', replace=True):
        """Merges measures and results from ``other`` study.
    
        :param other: The study which is to be merged into this study.
        :param sample_ids: The identifiers of the samples which are to be merged (or ``'all'``).
        :param replace: Whether conflicting identifiers are to be replaced (``True``) or prohibited (``False``).
        """
        for measure_name in other.measures:
            if measure_name not in self.measures.keys():
                self.add_measure(other.measures[measure_name], name=measure_name)
            for sample_id in (other.results[measure_name].keys() if sample_ids == 'all' else sample_ids):
                assert replace or sample_id not in self.results[measure_name]
                self.results[measure_name][sample_id] = list(other.results[measure_name][sample_id])
                if sample_id not in self.sample_ids: self.sample_ids.append(sample_id)
        self.results_cache.clear()

    def add_measure(self, measure, name=None):
        """Adds a performance measure to this study.
        
        :param measure: The performance measure to be added.
        :param name: An arbitrary name which uniquely identifies the performance measure within this study. Uses ``measure.default_name()`` if ``None`` is given.
        """
        if not isinstance(measure, Measure): raise ValueError(f'measure must be a Measure object ({type(measure)}, {measure})')
        if name is None: name = measure.default_name()
        self.measures[name] = measure
        self.results [name] = {None: []}

    def reset(self):
        """Resets all results computed so far in this study.
        """
        for measure_name in self.measures:
            self.results[measure_name] = {None: []}
        self.results_cache.clear()
        self.sample_ids.clear()

    def set_expected(self, expected, unique=True):
        """Sets the expected ground truth segmentation result.
        
        The background of the image must be labeled as ``0``. Negative object labels are forbidden. If ``unique`` is ``True``, it is assumed that all objects are labeled uniquely. Use ``unique=False`` if this is not guaranteed and the individual objects should be determined by connected component analysis instead  (e.g., if ``expected`` is a binary image which represents the union of the individual object masks).

        The image ``expected`` must be a numpy array of integral data type. It is also allowed to be boolean if and only if ``unique=False`` is used.

        :param expected: An image containing object masks corresponding to the ground truth.
        :param unique: Whether the individual object masks are uniquely labeled. Providing ``False`` assumes that connected components correspond to individual objects (components of different labels are not connected).
        """
        assert expected.min() == 0, 'mis-labeled ground truth'
        expected = expected.squeeze()
        assert expected.ndim == 2, 'ground truth has wrong dimensions'
        expected = _get_labeled(expected, unique, 'ground truth')
        for measure_name in self.measures:
            measure = self.measures[measure_name]
            measure.set_expected(expected)

    def process(self, sample_id, actual, unique=True, replace=True):
        """Evaluates a segmentation result based on the previously set expected result.
        
        If ``unique`` is ``True``, it is assumed that all objects are labeled uniquely. Use ``unique=False`` if this is not guaranteed and the individual objects should be determined by connected component analysis instead (e.g., if ``actual`` is a binary image which represents the union of the individual object masks).

        The image ``actual`` must be a numpy array of integral data type. It is also allowed to be boolean if and only if ``unique=False`` is used.

        :param sample_id: An arbitrary indentifier of the segmentation image (e.g., the file name).
        :param actual: An image containing object masks corresponding to the segmentation result.
        :param unique: Whether the individual object masks are uniquely labeled. Providing ``False`` assumes that connected components correspond to individual objects (components of different labels are not connected).
        :param replace: Whether previous results computed for the same ``sample_id`` should be replaced (``True``) or forbidden (``False``).
        """
        actual = actual.squeeze()
        assert actual.ndim == 2, 'image has wrong dimensions'
        actual = _get_labeled(actual, unique, 'image')
        assert replace or sample_id not in self.sample_ids
        intermediate_results = {}
        for measure_name in self.measures:
            measure = self.measures[measure_name]
            result = measure.compute(actual)
            self.results[measure_name][sample_id] = result
            intermediate_results[measure_name] = result
        self.results_cache.clear()
        self.sample_ids.append(sample_id)
        return intermediate_results

    def __getitem__(self, measure):
        """Returns list of all values recorded for ``measure``.
        """
        if measure not in self.results_cache:
            self.results_cache[measure] = list(itertools.chain(*[self.results[measure][sample_id] for sample_id in self.results[measure]]))
        return self.results_cache[measure]

    def print_results(self, write=sys.stdout.write, pad=0, fmt_unbound_float='g', line_suffix='\n'):
        """Prints the results of this study.
        
        :param write: Function used for output.
        :param pad: Number of whitespaces at the start of each line.
        :param fmt_unbound_float: Formatting literal used for decimals not presented as percentages.
        :param line_suffix: Suffix of each line.
        """
        label_length   = pad + max(len(str(measure_name)) for measure_name in self.results)
        fmt_fractional = '%%%ds: %%5.2f %%%%' % label_length
        fmt_unbound    = '%%%ds: %%%s' % (label_length, fmt_unbound_float)
        for measure_name in sorted(self.results.keys()):
            measure = self.measures[measure_name]
            fmt = fmt_fractional if measure.FRACTIONAL else fmt_unbound
            val = _aggregate(measure, self[measure_name]) * (100 if measure.FRACTIONAL else 1)
            write((fmt % (measure_name, val)) + line_suffix)

    def write_csv(self, fout, write_samples='auto', write_header=True, write_summary=True, **kwargs):
        """Writes the results of this study as CSV.
        
        :param fout: File descriptor used for output.
        :param write_samples: Whether all samples should be written separately.
        :param write_header: Whether a header should be included (measure names).
        :param write_summary: Whether a summary should be included in the last row.
        :param kwargs: Additional parameters for ``csv.writer``, see: https://docs.python.org/3/library/csv.html#csv.writer
        """
        kwargs.setdefault('delimiter', ',')
        kwargs.setdefault('quotechar', '"')
        kwargs.setdefault('quoting', csv.QUOTE_MINIMAL)
        rows = list()

        # define header
        if write_header:
            rows += [[self.csv_sample_id_column_name] + [measure_name for measure_name in self.measures.keys()]]

        # define samples
        if write_samples == True or (write_samples == 'auto' and len(self.sample_ids) > 1):
            for sample_id in sorted(self.sample_ids):
                row = [sample_id]
                for measure_name in self.measures.keys():
                    measure = self.measures[measure_name]
                    samples = self.results[measure_name]
                    row += [_aggregate(measure, samples[sample_id])]
                rows.append(row)

        # define summary
        if write_summary:
            rows.append([''])
            for measure_name in self.measures.keys():
                measure = self.measures[measure_name]
                value = _aggregate(measure, self[measure_name])
                rows[-1].append(value)

        # write results
        csv_writer = csv.writer(fout, **kwargs)
        for row in rows:
            csv_writer.writerow(row)

    def write_tsv(self, fout, **kwargs):
        """Writes the results of this study as TSV.
        
        :param fout: File descriptor used for output.
        :param kwargs: Additional parameters passed to :meth:`write_csv`.
        """
        kwargs.setdefault('delimiter', '\t')
        return self.write_csv(fout, **kwargs)

    def todf(self):
        """Returns the results of this study as a pandas dataframe.
        """
        import pandas as pd
        import io
        buf = io.StringIO()
        self.write_csv(buf, delimiter=',')
        buf.seek(0)
        df = pd.read_csv(buf, sep=',', keep_default_na=False)
        return df


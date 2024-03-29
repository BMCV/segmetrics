# flake8: noqa

import os
import pathlib
import tempfile
import time
import unittest
import warnings

import numpy as np
import pandas as pd
import skimage.io

import segmetrics as sm
from tests.data import (
    CrossSampler,
    images,
)
from tests.isbi_seg import isbi_seg_official


def create_full_study():
    study = sm.Study()
    study.add_measure(sm.Dice(), 'Dice')
    study.add_measure(sm.Dice().object_based(), 'Ob. Dice')
    study.add_measure(sm.Dice().object_based().reversed(), 'Rev. Ob. Dice')
    study.add_measure(sm.Dice().object_based().symmetric(), 'Sym. Ob. Dice')
    study.add_measure(sm.ISBIScore(), 'SEG')
    study.add_measure(sm.ISBIScore().reversed(), 'Rev. SEG')
    study.add_measure(sm.ISBIScore().symmetric(), 'Sym. SEG')
    study.add_measure(sm.JaccardCoefficient(), 'JC')
    study.add_measure(sm.JaccardIndex(), 'JI')
    study.add_measure(sm.JaccardIndex(aggregation='geometric-mean'), 'JI (geom)')
    study.add_measure(sm.RandIndex(), 'Rand')
    study.add_measure(sm.AdjustedRandIndex(), 'ARI')
    study.add_measure(sm.Hausdorff(), 'HSD')
    study.add_measure(sm.Hausdorff(quantile=0.99), 'QHSD')
    study.add_measure(sm.NSD(), 'NSD')
    study.add_measure(sm.Hausdorff().object_based(), 'Ob. HSD')
    study.add_measure(sm.Hausdorff().object_based().reversed(), 'Rev. Ob. HSD')
    study.add_measure(sm.Hausdorff().object_based().symmetric(), 'Sym. Ob. HSD')
    study.add_measure(sm.NSD().object_based(), 'Ob. NSD')
    study.add_measure(sm.FalseSplit(), 'Split')
    study.add_measure(sm.FalseMerge(), 'Merge')
    study.add_measure(sm.FalsePositive(), 'FP')
    study.add_measure(sm.FalseNegative(), 'FN')
    study.add_measure(sm.FalseSplit(aggregation='object-mean'), 'Split/obj')
    study.add_measure(sm.FalseMerge(aggregation='object-mean'), 'Merge/obj')
    return study


def compare_study(test, study, *args, **kwargs):
    compare_dataframe(test, study.todf(), *args, **kwargs)


def compare_dataframe(test, study_df, expected_csv_filepath, tag=None):
    expected_csv_filepath = pathlib.Path(expected_csv_filepath)
    actual_csv_filepath = f'{expected_csv_filepath}-out-{tag}' if tag else f'{expected_csv_filepath}-out'
    failure_message = f'Obtained results written to: {actual_csv_filepath}'
    try:
        test.assertTrue(expected_csv_filepath.is_file(), failure_message)
        expected_df = pd.read_csv(str(expected_csv_filepath), sep=',', keep_default_na=False)
        test.assertTrue(study_df.round(3).equals(expected_df.round(3)), failure_message)
    except:
        study_df.to_csv(actual_csv_filepath, index=False)
        raise


class MeasureTest(unittest.TestCase):

    def test_default_name(self):
        self.assertEqual(sm.Dice().object_based().default_name(), 'Ob. Dice')
        self.assertEqual(sm.Dice().object_based().reversed().default_name(), 'Rev. Ob. Dice')
        self.assertEqual(sm.Dice().object_based().symmetric().default_name(), 'Sym. Ob. Dice')
        self.assertEqual(sm.ISBIScore().default_name(), 'SEG')
        self.assertEqual(sm.ISBIScore().reversed().default_name(), 'Rev. SEG')
        self.assertEqual(sm.ISBIScore().symmetric().default_name(), 'Sym. SEG')
        self.assertEqual(sm.JaccardCoefficient().default_name(), 'Jaccard coef.')
        self.assertEqual(sm.JaccardIndex().default_name(), 'Jaccard index')
        self.assertEqual(sm.RandIndex().default_name(), 'Rand')
        self.assertEqual(sm.AdjustedRandIndex().default_name(), 'ARI')
        self.assertEqual(sm.Hausdorff().default_name(), 'HSD')
        self.assertEqual(sm.Hausdorff(quantile=0.99).default_name(), 'HSD (Q=0.99)')
        self.assertEqual(sm.NSD().default_name(), 'NSD')
        self.assertEqual(sm.Hausdorff().object_based().default_name(), 'Ob. HSD')
        self.assertEqual(sm.Hausdorff().object_based().reversed().default_name(), 'Rev. Ob. HSD')
        self.assertEqual(sm.Hausdorff().object_based().symmetric().default_name(), 'Sym. Ob. HSD')
        self.assertEqual(sm.NSD().object_based().default_name(), 'Ob. NSD')
        self.assertEqual(sm.FalseSplit().default_name(), 'Split')
        self.assertEqual(sm.FalseMerge().default_name(), 'Merge')
        self.assertEqual(sm.FalsePositive().default_name(), 'Spurious')
        self.assertEqual(sm.FalseNegative().default_name(), 'Missing')


class ObjMeanTest(unittest.TestCase):

    def do_test(self, measure):
        objects = 0
        study = sm.Study()
        sampler = CrossSampler(images, images)
        measure_name = study.add_measure(measure(aggregation='object-mean'))
        for sample_id, ref, seg in sampler.all():
            study.set_expected(ref, unique=True)
            objects += len(frozenset(ref.reshape(-1)) - frozenset([0]))
            study.process(sample_id, seg, unique=True)
        count = np.sum(study[measure_name])
        df = study.todf()
        expected = count / objects
        actual = df[measure_name].values[-1]
        self.assertAlmostEqual(actual, expected)

    def test_FalseMerge(self):
        self.do_test(sm.FalseMerge)

    def test_FalseSplit(self):
        self.do_test(sm.FalseSplit)


class FullStudyTest(unittest.TestCase):

    def setUp(self):
        self.started_at = time.time()
        self.study   = create_full_study()
        self.sampler = CrossSampler(images, images)

    def tearDown(self):
        elapsed = time.time() - self.started_at
        FullStudyTest.times[self.id()] = round(elapsed, 2)

    def test_sequential(self):
        for sample_id, ref, seg in self.sampler.all():
            with self.subTest(sample_id=sample_id):
                self.study.set_expected(ref, unique=True)
                self.study.process(sample_id, seg, unique=True)
        compare_study(self, self.study, 'tests/full-study-test.csv', 'sequential')

    def test_parallel(self):
        sm.parallel.process_all(self.study, lambda sid: self.sampler.img2(sid), lambda sid: self.sampler.img1(sid), self.sampler.sample_ids, num_forks=2, is_actual_unique=True, is_expected_unique=True)
        compare_study(self, self.study, 'tests/full-study-test.csv', 'parallel')

    @classmethod
    def setUpClass(cls):
        cls.times = dict()

    @classmethod
    def tearDownClass(cls):
        print('\nPerformance:')
        for test_id, duration in cls.times.items():
            print(f'  {test_id}: {duration} sec')


class SEGTest(unittest.TestCase):

    def setUp(self):
        self.study = sm.Study()
        self.study.add_measure(sm.ISBIScore(), 'SEG')
        self.sampler = CrossSampler(images, images)

    def test_parallel(self):
        sm.parallel.process_all(self.study, lambda sid: self.sampler.img2(sid), lambda sid: self.sampler.img1(sid), self.sampler.sample_ids, num_forks=2, is_actual_unique=True, is_expected_unique=True)
        seg_expected = isbi_seg_official(self.sampler.img2_list, self.sampler.img1_list)
        seg_actual = np.mean(self.study['SEG'])
        error = abs(seg_actual - seg_expected)
        self.assertTrue(error < 1e-5, f'Expected {seg_expected}, but got {seg_actual} (error: {error}')


class CLITest(unittest.TestCase):

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tempdir:
            segdir = tempdir + '/seg'
            os.mkdir(segdir)
            for img_num, image in enumerate(images, start=1):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    skimage.io.imsave(f'{segdir}/img{img_num}.png', image)
            with tempfile.NamedTemporaryFile(suffix='.csv') as result_file:
                os.system(fr'python -m segmetrics.cli {segdir} ".*img([0-9]+).png" {segdir}/img\\1.png {result_file.name} "sm.Dice()" "sm.ISBIScore()" "sm.FalseMerge()" "sm.FalseSplit()" >/dev/null')
                actual_df = pd.read_csv(result_file.name, sep=',', keep_default_na=False)
            compare_dataframe(self, actual_df, 'tests/cli-test.csv')


if __name__ == '__main__':
    unittest.main()


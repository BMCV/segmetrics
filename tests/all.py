import unittest
import segmetrics as sm
import numpy as np
import pathlib
import pandas as pd

from tests.data import images, CrossSampler


def create_full_study():
    study = sm.Study()
    study.add_measure(sm.Dice(), 'Dice')
    study.add_measure(sm.ISBIScore(), 'SEG')
    study.add_measure(sm.JaccardCoefficient(), 'JC')
    study.add_measure(sm.JaccardIndex(), 'JI')
    study.add_measure(sm.RandIndex(), 'Rand')
    study.add_measure(sm.AdjustedRandIndex(), 'ARI')
    study.add_measure(sm.Hausdorff('sym'), 'HSD (sym)')
    study.add_measure(sm.Hausdorff('e2a'), 'HSD (e2a)')
    study.add_measure(sm.Hausdorff('a2e'), 'HSD (a2e)')
    study.add_measure(sm.NSD(), 'NSD')
    study.add_measure(sm.object_based_distance(sm.Hausdorff('sym')), 'Ob. HSD (sym)')
    study.add_measure(sm.object_based_distance(sm.Hausdorff('e2a')), 'Ob. HSD (e2a)')
    study.add_measure(sm.object_based_distance(sm.Hausdorff('a2e')), 'Ob. HSD (a2e)')
    study.add_measure(sm.object_based_distance(sm.NSD()), 'Ob. NSD')
    study.add_measure(sm.FalseSplit(), 'Split')
    study.add_measure(sm.FalseMerge(), 'Merge')
    study.add_measure(sm.FalsePositive(), 'FP')
    study.add_measure(sm.FalseNegative(), 'FN')
    return study


def compare_study(test, study, expected_csv_filepath):
    expected_csv_filepath = pathlib.Path(expected_csv_filepath)
    try:
        study_df = study.todf().round(3)
        test.assertTrue(expected_csv_filepath.is_file())
        expected_df = pd.read_csv(str(expected_csv_filepath), sep=',', keep_default_na=False).round(3)
        test.assertTrue(study_df.equals(expected_df))
    except:
        actual_csv_filepath = f'{expected_csv_filepath}-out'
        study_df.to_csv(actual_csv_filepath, index=False)
        print(f'Obtained results written to: {actual_csv_filepath}')
        raise


class FullStudyTest(unittest.TestCase):

    def setup(self):
        self.study   = create_full_study()
        self.sampler = CrossSampler(images, images)

    def test_sequential(self):
        self.setup()
        for sample_id, ref, seg in self.sampler.all():
            with self.subTest(sample_id=sample_id):
                self.study.set_expected(ref, unique=True)
                self.study.process(sample_id, seg, unique=True)
        compare_study(self, self.study, 'tests/full-study-test.csv')

    #def parallel(self):
    #    from tests.data import images
    #    study = create_full_study()
    #    pass


if __name__ == '__main__':
    unittest.main()


import argparse
import re
import glob
import pathlib
import skimage.io

import segmetrics as sm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('seg_dir', help='directory containing the segmentation results')
    parser.add_argument('seg_file', help='regex pattern used to parse segmentation results')
    parser.add_argument('gt_file', help='pattern used to obtain the corresponding ground truth file (\\1 corresponds to the first capture group)')
    parser.add_argument('output_file', help='filepath where the results are to be written (CSV)')
    parser.add_argument('--recursive', '-r', action='store_true', help='reads the segmentation results directory recursively')
    parser.add_argument('--gt-unique', action='store_true', help='assumes that the ground truth data is uniquely labeled')
    parser.add_argument('--seg-unique', action='store_true', help='assumes that the segmentation result data is uniquely labeled')
    parser.add_argument('measures', nargs='+', type=str, help='list of performance measures')
    args = parser.parse_args()

    measure_spec_pattern = re.compile(r'([a-zA-Z]+)((:?_o)?)')

    print(f'Summary')
    print(f'*******')
    print(f'')
    print(f'  Reading segmentation results from: {args.seg_dir}')
    print(f'  Pattern used for parsing file name: {args.seg_file}')
    print(f'  Pattern used for ground truth files: {args.gt_file}')
    print(f'  Results will be written to: {args.output_file}')
    print(f'  The following performance measures will be used:')

    study = sm.Study()
    for measure_spec in args.measures:
        print(f'  - {measure_spec}')
        measure = eval(measure_spec)
        study.add_measure(measure)

    seg_file_pattern = re.compile(args.seg_file)

    print(f'')
    print(f'Evaluation')
    print(f'**********')
    print(f'')

    for filepath in glob.glob(args.seg_dir + '**' if args.seg_dir.endswith('/') else args.seg_dir + '/**', recursive=args.recursive):
        match = seg_file_pattern.match(filepath)
        if match is None: continue
        gt_file = args.gt_file
        for group_idx in range(len(match.groups()) + 1):
            gt_file = gt_file.replace(rf'\{group_idx:d}', match.group(group_idx))

        print(f'Evaluating {filepath} using ground truth: {gt_file}')

        sample_id = str(pathlib.Path(filepath).relative_to(args.seg_dir))

        im_actual   = skimage.io.imread(filepath)
        im_expected = skimage.io.imread(gt_file)

        study.set_expected(im_expected, unique=args.gt_unique)
        study.process(sample_id, im_actual, unique=args.seg_unique)

    with open(args.output_file, 'w') as fout:
        study.write_csv(fout)

    print(f'')
    print(f'Results written to: {args.output_file}')


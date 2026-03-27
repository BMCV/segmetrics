import os
import re
import subprocess
import tempfile
import warnings

import py7zr
from skimage import io


def isbi_seg_official(result_list, groundtruth_list, verbose=False):
    assert len(groundtruth_list) == len(result_list), 'data mismatch'
    subprocess_kwargs = dict() if verbose else dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        os.makedirs('data/01_GT/SEG')
        os.makedirs('data/01_RES')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, data in enumerate(zip(groundtruth_list, result_list)):
                groundtruth, result = data
                io.imsave('data/01_RES/mask%03d.tif'       % i, result     [None, :, :].astype('uint16'), plugin='tifffile')
                io.imsave('data/01_GT/SEG/man_seg%03d.tif' % i, groundtruth[None, :, :].astype('uint16'), plugin='tifffile')

        subprocess.call(['wget', 'http://evoid.de/isbi_evaluation_software.7z'], **subprocess_kwargs)
        with py7zr.SevenZipFile('isbi_evaluation_software.7z', mode='r', password='ppy42wGfcrHG9W4Z') as z:
            z.extractall()
        subprocess.call(['chmod', '+w', '-R', '.'], **subprocess_kwargs)

        os.chdir('Linux')
        subprocess.call(['chmod', '+x', 'SEGMeasure'], **subprocess_kwargs)
        result = str(subprocess.check_output(['./SEGMeasure', '../data', '01']))

        match = re.compile(r'.*?([\.0-9]+)').match(result)
        if match is not None:
            official_result = float(match.group(1))
            return official_result
        else:
            raise ValueError('Unexpected result:', result)

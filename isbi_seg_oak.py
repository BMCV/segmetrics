import os
import tempfile
import shutil
import subprocess
import sys
from skimage import io

def oak(groundtruth_list, result_list):
    assert len(groundtruth_list) == len(result_list), 'data mismatch'
    tmp_dir = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(tmp_dir)
        os.makedirs('data/01_GT/SEG')
        os.makedirs('data/01_RES')
        
        for i, data in enumerate(zip(groundtruth_list, result_list)):
            groundtruth, result = data
            io.imsave('data/01_RES/mask%03d.tif'       % i, result     [None, :, :].astype('uint16'), plugin='tifffile')
            io.imsave('data/01_GT/SEG/man_seg%03d.tif' % i, groundtruth[None, :, :].astype('uint16'), plugin='tifffile')
        
        subprocess.call(['wget', 'http://ctc2015.gryf.fi.muni.cz/Public/Software/EvaluationSoftware.zip'])
        subprocess.call(['unzip', '*.zip'])
        subprocess.call(['chmod', '+w', '-R', '.'])
        
        os.chdir('Linux')
        subprocess.call(['chmod', '+x', 'SEGMeasure'])
        return subprocess.check_output(['./SEGMeasure', '../data', '01'])
        
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp_dir)


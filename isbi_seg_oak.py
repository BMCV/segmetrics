import os
import tempfile
import shutil
import subprocess
import sys
from skimage import io

def oak(groundtruth, result):
    tmp_dir = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(tmp_dir)
        os.makedirs('data/01_GT/SEG')
        os.makedirs('data/01_RES')
    
        io.imsave('data/01_RES/mask000.tif'      , result     [None, :, :].astype('uint16'), plugin='tifffile')
        io.imsave('data/01_GT/SEG/man_seg000.tif', groundtruth[None, :, :].astype('uint16'), plugin='tifffile')
        
        subprocess.call(['wget', 'http://ctc2015.gryf.fi.muni.cz/Public/Software/EvaluationSoftware.zip'])
        subprocess.call(['unzip', '*.zip'])
        subprocess.call(['chmod', '+w', '-R', '.'])
        
        os.chdir('Linux')
        subprocess.call(['chmod', '+x', 'SEGMeasure'])
        return subprocess.check_output(['./SEGMeasure', '../data', '01'])
        
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp_dir)


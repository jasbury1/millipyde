import sys
import pathlib
cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))

import unittest

import numpy as np
import numpy.testing as npt
from skimage import data, io, filters, transform
from skimage.io import imsave, imread
from skimage.color import rgb2gray, rgba2rgb

import millipyde as mp
from millipyde import Operation

DECIMAL_ERROR = 4

class TestMillypdeMultiGPU(unittest.TestCase):

    def test_sample(self):
        self.assertTrue(True)
    


if __name__ == '__main__':
    unittest.main()

import sys
sys.path.append('/home/jasbury/Thesis/millipyde')

import unittest

import numpy as np
from skimage import data, io, filters, transform
from skimage.io import imsave, imread
from skimage.color import rgb2gray, rgba2rgb

import millipyde as mp

class TestMillipydeImages(unittest.TestCase):

    def test_sample(self):
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()
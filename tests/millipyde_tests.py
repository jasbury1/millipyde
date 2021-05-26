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

DECIMAL_ERROR = 4

class TestMillipydeImages(unittest.TestCase):

    def test_create_gpuarray(self):
        numpy_array = np.array([1, 2, 3, 4])
        gpu_array = mp.gpuarray(numpy_array)
        self.assertIsNotNone(gpu_array)
        numpy_array2 = np.array(gpu_array)
        self.assertTrue(np.array_equal(numpy_array, numpy_array2))
    

    def test_create_gpuarray2(self):
        numpy_array = np.array([1, 2, 3, 4])
        gpu_array = mp.gpuarray([1, 2, 3, 4])
        self.assertTrue(np.array_equal(numpy_array, np.array(gpu_array)))


    def test_create_gpuarray2(self):
        numpy_array = np.array([1, 2, 3, 4])
        gpu_array = mp.gpuarray([4, 3, 2, 1])
        self.assertFalse(np.array_equal(numpy_array, np.array(gpu_array)))
    

    def test_create_invalid_gpuarray(self):
        with self.assertRaises(ValueError):
            gpu_array = mp.gpuarray(None)


    def test_create_invalid_gpuarray2(self):
        with self.assertRaises(TypeError):
            gpu_array = mp.gpuarray()


    def test_open_image(self):
        charlie = io.imread("examples/images/charlie.png")
        charlie_on_gpu = mp.gpuarray(io.imread("examples/images/charlie.png"))
        charlie2 = np.array(charlie_on_gpu)
        npt.assert_almost_equal(charlie, charlie2, decimal=DECIMAL_ERROR)


    def test_rgb2grey(self):
        charlie = io.imread("examples/images/charlie.png")
        grey_charlie = rgb2gray(rgba2rgb(charlie))

        charlie_on_gpu = mp.gpuarray(io.imread("examples/images/charlie.png"))
        charlie_on_gpu.rgb2grey()
        grey_charlie2 = np.array(charlie_on_gpu)

        npt.assert_almost_equal(grey_charlie, grey_charlie2, decimal=DECIMAL_ERROR)


    def test_transpose(self):
        charlie = io.imread("examples/images/charlie.png")
        charlie = np.transpose(rgb2gray(rgba2rgb(charlie)))

        charlie2 = mp.gpuarray(io.imread("examples/images/charlie.png"))
        charlie2.rgb2grey()
        charlie2.transpose()
        charlie2 = np.array(charlie2)

        npt.assert_almost_equal(charlie, charlie2, decimal=DECIMAL_ERROR)



if __name__ == '__main__':
    unittest.main()

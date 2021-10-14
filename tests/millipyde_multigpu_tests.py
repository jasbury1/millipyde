from re import A
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

    def test_device_count(self):
        num_devices = mp.device_count()
        self.assertEqual(2, num_devices)


    def test_device_count2(self):
        self.assertEqual(2, mp.DEVICE_COUNT)


    def test_device_handoff(self):
        charlie = io.imread("examples/images/charlie.png")
        charlie = rgb2gray(rgba2rgb(charlie))
        with mp.Device(0):
            d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))
            self.assertEqual(0, mp.get_current_device())
            with mp.Device(1):
                d_charlie.rgb2grey()
                self.assertEqual(1, mp.get_current_device())
        
                npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)


    def test_device_handoff2(self):
        charlie = io.imread("examples/images/charlie.png")
        charlie = rgb2gray(rgba2rgb(charlie))
        with mp.Device(0):
            d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))
            self.assertEqual(0, mp.get_current_device())
            with mp.Device(1):
                d_charlie.rgb2grey()
                self.assertEqual(1, mp.get_current_device())
            self.assertEqual(0, mp.get_current_device()) 
            npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)


    def test_device_handoff3(self):
        charlie = io.imread("examples/images/charlie.png")
        charlie = rgb2gray(rgba2rgb(charlie))
        with mp.Device(0):
            d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))
            self.assertEqual(0, mp.get_current_device())
            with mp.Device(1):
                d_charlie.rgb2grey()
                self.assertEqual(1, mp.get_current_device())
        npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)

    
    def test_dual_pipelines(self):
        charlie = io.imread("examples/images/charlie.png")
        charlie = np.transpose(rgb2gray(rgba2rgb(charlie)))

        d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))

        inputs = [d_charlie]
        operations = [mp.Operation("rgb2grey")]
        operations2 = [mp.Operation("transpose")]

        p = mp.Pipeline(inputs, operations, device=0)
        p2 = mp.Pipeline([], operations2, device=1)
        p.connect_to(p2)
        p.run()

        npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)


    def test_dual_pipelines_unspecified_devices(self):
        charlie = io.imread("examples/images/charlie.png")
        charlie = np.transpose(rgb2gray(rgba2rgb(charlie)))

        d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))

        inputs = [d_charlie]
        operations = [mp.Operation("rgb2grey")]
        operations2 = [mp.Operation("transpose")]

        p = mp.Pipeline(inputs, operations)
        p2 = mp.Pipeline([], operations2)
        p.connect_to(p2)
        p.run()

        npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)

    
    def test_dual_pipelines_unspecified_devices2(self):
        charlie = io.imread("examples/images/charlie.png")
        charlie = np.transpose(np.transpose(rgb2gray(rgba2rgb(charlie))))

        d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))

        inputs = [d_charlie]
        operations = [mp.Operation("rgb2grey")]
        operations2 = [mp.Operation("transpose")]
        operations3 = [mp.Operation("transpose")]

        p = mp.Pipeline(inputs, operations)
        p2 = mp.Pipeline([], operations2)
        p3 = mp.Pipeline([], operations3)
        p.connect_to(p2)
        p2.connect_to(p3)
        p.run()

        npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)


 


if __name__ == '__main__':
    unittest.main()

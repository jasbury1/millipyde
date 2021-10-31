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
        charlie = io.imread("tests/images/charlie.png")
        charlie = rgb2gray(rgba2rgb(charlie))
        with mp.Device(0):
            d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
            self.assertEqual(0, mp.get_current_device())
            with mp.Device(1):
                d_charlie.rgb2grey()
                self.assertEqual(1, mp.get_current_device())
        
                npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)


    def test_device_handoff2(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = rgb2gray(rgba2rgb(charlie))
        with mp.Device(0):
            d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
            self.assertEqual(0, mp.get_current_device())
            with mp.Device(1):
                d_charlie.rgb2grey()
                self.assertEqual(1, mp.get_current_device())
            self.assertEqual(0, mp.get_current_device()) 
            npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)


    def test_device_handoff3(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = rgb2gray(rgba2rgb(charlie))
        with mp.Device(0):
            d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
            self.assertEqual(0, mp.get_current_device())
            with mp.Device(1):
                d_charlie.rgb2grey()
                self.assertEqual(1, mp.get_current_device())
        npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)

    
    def test_dual_pipelines(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = np.transpose(rgb2gray(rgba2rgb(charlie)))

        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))

        inputs = [d_charlie]
        operations = [mp.Operation("rgb2grey")]
        operations2 = [mp.Operation("transpose")]

        p = mp.Pipeline(inputs, operations, device=0)
        p2 = mp.Pipeline([], operations2, device=1)
        p.connect_to(p2)
        p.run()

        npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)


    def test_dual_pipelines_unspecified_devices(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = np.transpose(rgb2gray(rgba2rgb(charlie)))

        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))

        inputs = [d_charlie]
        operations = [mp.Operation("rgb2grey")]
        operations2 = [mp.Operation("transpose")]

        p = mp.Pipeline(inputs, operations)
        p2 = mp.Pipeline([], operations2)
        p.connect_to(p2)
        p.run()

        npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=DECIMAL_ERROR)

    
    def test_dual_pipelines_unspecified_devices2(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = np.transpose(np.transpose(rgb2gray(rgba2rgb(charlie))))

        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))

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

    
    def test_long_pipeline(self):
        d_charlie_control = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie_control.gaussian(2)
        d_charlie_control.rgb2grey()
        d_charlie_control.transpose()
        d_charlie_control.transpose()
        d_charlie_control.rotate(45)
        d_charlie_control = np.array(d_charlie_control)
        
        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie2 = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie3 = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie4 = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie5 = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie6 = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie7 = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie8 = mp.gpuimage(io.imread("tests/images/charlie.png"))

        charlies = [d_charlie, d_charlie2, d_charlie3, d_charlie4,
                    d_charlie5, d_charlie6, d_charlie7, d_charlie8]
        operations = [
            mp.Operation("gaussian", 2),
            mp.Operation("rgb2grey"),
            mp.Operation("transpose"),
            mp.Operation("transpose"),
            mp.Operation("rotate", 45)
        ]

        p = mp.Pipeline(charlies, operations)
        p.run()

        npt.assert_almost_equal(d_charlie_control, np.array(
            d_charlie), decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(d_charlie_control, np.array(
            d_charlie2), decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(d_charlie_control, np.array(
            d_charlie3), decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(d_charlie_control, np.array(
            d_charlie4), decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(d_charlie_control, np.array(
            d_charlie5), decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(d_charlie_control, np.array(
            d_charlie6), decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(d_charlie_control, np.array(
            d_charlie7), decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(d_charlie_control, np.array(
            d_charlie8), decimal=DECIMAL_ERROR)


    def test_clone(self):
        h_charlie = io.imread("tests/images/charlie.png")
        h_charlie2 = io.imread("tests/images/charlie.png")
        h_charlie2 = rgb2gray(rgba2rgb(h_charlie2))

        with mp.Device(0):
            d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))

        with mp.Device(1):
            d_charlie2 = d_charlie.clone()

        d_charlie2.rgb2grey()

        npt.assert_almost_equal(h_charlie, np.array(d_charlie),
                                decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(h_charlie2, np.array(d_charlie2),
                                decimal=DECIMAL_ERROR)


    def test_clone2(self):
        h_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        h_array2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        with mp.Device(0): 
            d_array = mp.gpuarray(h_array)
        with mp.Device(1):
            d_array2 = d_array.clone()
        
        npt.assert_almost_equal(h_array, np.array(d_array),
                                decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(h_array2, np.array(d_array2),
                                decimal=DECIMAL_ERROR)


    def test_generator(self):
        o = mp.Operation("rgb2grey")
        
        with mp.Device(1):
            g = mp.Generator("examples/images", [o])

        aspen = rgb2gray(rgba2rgb(io.imread("tests/images/aspen.png")))
        aspen2 = rgb2gray(rgba2rgb(io.imread("tests/images/aspen2.png")))
        charlie_small = rgb2gray(rgba2rgb(io.imread("tests/images/charlie_small.png")))
        charlie = rgb2gray(rgba2rgb(io.imread("tests/images/charlie.png")))
        charlie2 = rgb2gray(rgba2rgb(io.imread("tests/images/charlie2.png")))
        double_trouble = rgb2gray(rgba2rgb(io.imread("tests/images/double_trouble.png")))

        for i in range(14):
            out = np.array(next(g))
            if i % 6 == 0:
                npt.assert_almost_equal(out, aspen2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 1:
                npt.assert_almost_equal(out, aspen,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 2:
                npt.assert_almost_equal(out, charlie,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 3:
                npt.assert_almost_equal(out, charlie_small,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 4:
                npt.assert_almost_equal(out, charlie2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 5:
                npt.assert_almost_equal(out, double_trouble,
                                        decimal=DECIMAL_ERROR)
        
    def test_generator2(self):
        o = mp.Operation("rgb2grey")
    
        g = mp.Generator("examples/images", [o])

        aspen = rgb2gray(rgba2rgb(io.imread("tests/images/aspen.png")))
        aspen2 = rgb2gray(rgba2rgb(io.imread("tests/images/aspen2.png")))
        charlie_small = rgb2gray(rgba2rgb(io.imread("tests/images/charlie_small.png")))
        charlie = rgb2gray(rgba2rgb(io.imread("tests/images/charlie.png")))
        charlie2 = rgb2gray(rgba2rgb(io.imread("tests/images/charlie2.png")))
        double_trouble = rgb2gray(rgba2rgb(io.imread("tests/images/double_trouble.png")))

        with mp.Device(1):
            for i in range(14):
                out = np.array(next(g))
                if i % 6 == 0:
                    npt.assert_almost_equal(out, aspen2,
                                            decimal=DECIMAL_ERROR)
                if i % 6 == 1:
                    npt.assert_almost_equal(out, aspen,
                                            decimal=DECIMAL_ERROR)
                if i % 6 == 2:
                    npt.assert_almost_equal(out, charlie,
                                            decimal=DECIMAL_ERROR)
                if i % 6 == 3:
                    npt.assert_almost_equal(out, charlie_small,
                                            decimal=DECIMAL_ERROR)
                if i % 6 == 4:
                    npt.assert_almost_equal(out, charlie2,
                                            decimal=DECIMAL_ERROR)
                if i % 6 == 5:
                    npt.assert_almost_equal(out, double_trouble,
                                            decimal=DECIMAL_ERROR)


    def test_generator3(self):
        o = mp.Operation("rgb2grey")
        
        with mp.Device(1):
            g = mp.Generator("examples/images", [o], device=1)

        aspen = rgb2gray(rgba2rgb(io.imread("tests/images/aspen.png")))
        aspen2 = rgb2gray(rgba2rgb(io.imread("tests/images/aspen2.png")))
        charlie_small = rgb2gray(rgba2rgb(io.imread("tests/images/charlie_small.png")))
        charlie = rgb2gray(rgba2rgb(io.imread("tests/images/charlie.png")))
        charlie2 = rgb2gray(rgba2rgb(io.imread("tests/images/charlie2.png")))
        double_trouble = rgb2gray(rgba2rgb(io.imread("tests/images/double_trouble.png")))

        for i in range(14):
            out = np.array(next(g))
            if i % 6 == 0:
                npt.assert_almost_equal(out, aspen2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 1:
                npt.assert_almost_equal(out, aspen,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 2:
                npt.assert_almost_equal(out, charlie,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 3:
                npt.assert_almost_equal(out, charlie_small,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 4:
                npt.assert_almost_equal(out, charlie2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 5:
                npt.assert_almost_equal(out, double_trouble,
                                        decimal=DECIMAL_ERROR) 


    def test_generator_return_to_host(self):
        o = mp.Operation("rgb2grey")
        g = mp.Generator("examples/images", [o], return_to_host=True)

        aspen = rgb2gray(rgba2rgb(io.imread("tests/images/aspen.png")))
        aspen2 = rgb2gray(rgba2rgb(io.imread("tests/images/aspen2.png")))
        charlie_small = rgb2gray(rgba2rgb(io.imread("tests/images/charlie_small.png")))
        charlie = rgb2gray(rgba2rgb(io.imread("tests/images/charlie.png")))
        charlie2 = rgb2gray(rgba2rgb(io.imread("tests/images/charlie2.png")))
        double_trouble = rgb2gray(rgba2rgb(io.imread("tests/images/double_trouble.png")))

        for i in range(7):
            out = next(g)
            if i % 6 == 0:
                npt.assert_almost_equal(out, aspen2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 1:
                npt.assert_almost_equal(out, aspen,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 2:
                npt.assert_almost_equal(out, charlie,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 3:
                npt.assert_almost_equal(out, charlie_small,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 4:
                npt.assert_almost_equal(out, charlie2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 5:
                npt.assert_almost_equal(out, double_trouble,
                                        decimal=DECIMAL_ERROR)


    def test_generator_return_to_host2(self):
        o = mp.Operation("rgb2grey")
        with mp.Device(1):
            g = mp.Generator("examples/images", [o], return_to_host=True)

        aspen = rgb2gray(rgba2rgb(io.imread("tests/images/aspen.png")))
        aspen2 = rgb2gray(rgba2rgb(io.imread("tests/images/aspen2.png")))
        charlie_small = rgb2gray(rgba2rgb(io.imread("tests/images/charlie_small.png")))
        charlie = rgb2gray(rgba2rgb(io.imread("tests/images/charlie.png")))
        charlie2 = rgb2gray(rgba2rgb(io.imread("tests/images/charlie2.png")))
        double_trouble = rgb2gray(rgba2rgb(io.imread("tests/images/double_trouble.png")))

        for i in range(7):
            out = next(g)
            if i % 6 == 0:
                npt.assert_almost_equal(out, aspen2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 1:
                npt.assert_almost_equal(out, aspen,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 2:
                npt.assert_almost_equal(out, charlie,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 3:
                npt.assert_almost_equal(out, charlie_small,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 4:
                npt.assert_almost_equal(out, charlie2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 5:
                npt.assert_almost_equal(out, double_trouble,
                                        decimal=DECIMAL_ERROR)

    def test_generator_return_to_host2(self):
        o = mp.Operation("rgb2grey")
        g = mp.Generator("examples/images", [o], return_to_host=True, device=1)

        aspen = rgb2gray(rgba2rgb(io.imread("tests/images/aspen.png")))
        aspen2 = rgb2gray(rgba2rgb(io.imread("tests/images/aspen2.png")))
        charlie_small = rgb2gray(rgba2rgb(io.imread("tests/images/charlie_small.png")))
        charlie = rgb2gray(rgba2rgb(io.imread("tests/images/charlie.png")))
        charlie2 = rgb2gray(rgba2rgb(io.imread("tests/images/charlie2.png")))
        double_trouble = rgb2gray(rgba2rgb(io.imread("tests/images/double_trouble.png")))

        for i in range(7):
            out = next(g)
            if i % 6 == 0:
                npt.assert_almost_equal(out, aspen2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 1:
                npt.assert_almost_equal(out, aspen,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 2:
                npt.assert_almost_equal(out, charlie,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 3:
                npt.assert_almost_equal(out, charlie_small,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 4:
                npt.assert_almost_equal(out, charlie2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 5:
                npt.assert_almost_equal(out, double_trouble,
                                        decimal=DECIMAL_ERROR)


    def test_generator_fixed_outputs(self):
        o = mp.Operation("rgb2grey")
        g = mp.Generator("examples/images", [o], return_to_host=True, outputs=3)

        aspen = rgb2gray(rgba2rgb(io.imread("tests/images/aspen.png")))
        aspen2 = rgb2gray(rgba2rgb(io.imread("tests/images/aspen2.png")))
        charlie_small = rgb2gray(rgba2rgb(io.imread("tests/images/charlie_small.png")))
        charlie = rgb2gray(rgba2rgb(io.imread("tests/images/charlie.png")))
        charlie2 = rgb2gray(rgba2rgb(io.imread("tests/images/charlie2.png")))
        double_trouble = rgb2gray(rgba2rgb(io.imread("tests/images/double_trouble.png")))

        i = 0
        for out in g:
            if i % 6 == 0:
                npt.assert_almost_equal(out, aspen2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 1:
                npt.assert_almost_equal(out, aspen,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 2:
                npt.assert_almost_equal(out, charlie,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 3:
                npt.assert_almost_equal(out, charlie_small,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 4:
                npt.assert_almost_equal(out, charlie2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 5:
                npt.assert_almost_equal(out, double_trouble,
                                        decimal=DECIMAL_ERROR)
            i += 1
        
        self.assertEqual(i, 3)

    
    def test_generator_fixed_outputs2(self):
        o = mp.Operation("rgb2grey")
        g = mp.Generator("examples/images", [o], return_to_host=True, outputs=3, device=1)

        aspen = rgb2gray(rgba2rgb(io.imread("tests/images/aspen.png")))
        aspen2 = rgb2gray(rgba2rgb(io.imread("tests/images/aspen2.png")))
        charlie_small = rgb2gray(rgba2rgb(io.imread("tests/images/charlie_small.png")))
        charlie = rgb2gray(rgba2rgb(io.imread("tests/images/charlie.png")))
        charlie2 = rgb2gray(rgba2rgb(io.imread("tests/images/charlie2.png")))
        double_trouble = rgb2gray(rgba2rgb(io.imread("tests/images/double_trouble.png")))

        i = 0
        for out in g:
            if i % 6 == 0:
                npt.assert_almost_equal(out, aspen2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 1:
                npt.assert_almost_equal(out, aspen,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 2:
                npt.assert_almost_equal(out, charlie,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 3:
                npt.assert_almost_equal(out, charlie_small,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 4:
                npt.assert_almost_equal(out, charlie2,
                                        decimal=DECIMAL_ERROR)
            if i % 6 == 5:
                npt.assert_almost_equal(out, double_trouble,
                                        decimal=DECIMAL_ERROR)
            i += 1
        
        self.assertEqual(i, 3)



if __name__ == '__main__':
    unittest.main()

import sys
import pathlib
cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))

import unittest

import numpy as np
import numpy.testing as npt
from skimage import data, io, filters, transform, exposure
from skimage.io import imsave, imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import gaussian

import millipyde as mp
from millipyde import Operation

DECIMAL_ERROR = 4

class TestMillipyde(unittest.TestCase):
    
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


    def test_create_gpuarray3(self):
        numpy_array = np.array([1, 2, 3, 4])
        gpu_array = mp.gpuarray([4, 3, 2, 1])
        self.assertFalse(np.array_equal(numpy_array, np.array(gpu_array)))
    
    
    def test_create_invalid_gpuarray(self):
        with self.assertRaises(ValueError):
            gpu_array = mp.gpuarray(None)
    

    def test_create_invalid_gpuarray2(self):
        with self.assertRaises(TypeError):
            gpu_array = mp.gpuarray()


    def test_create_gpuimage(self):
        numpy_array = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
        gpu_image = mp.gpuimage(numpy_array)
        self.assertIsNotNone(gpu_image)
        numpy_array2 = np.array(gpu_image)
        self.assertTrue(np.array_equal(numpy_array, numpy_array2))


    def test_create_gpuimage2(self):
        numpy_array = np.array([[[1, 2, 3], [4, 5, 6]], 
                                [[1, 2, 3], [4, 5, 6]]])
        gpu_image = mp.gpuimage(numpy_array)
        self.assertIsNotNone(gpu_image)
        numpy_array2 = np.array(gpu_image)
        self.assertTrue(np.array_equal(numpy_array, numpy_array2))


    def test_create_invalid_gpuimage(self):
        numpy_array = np.array([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            gpu_image = mp.gpuimage(numpy_array)


    def test_create_invalid_gpuimage2(self):
        numpy_array = np.array([[[[1, 2], [3, 4]], [[1, 2], [3, 4]]], 
                                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]], 
                                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]])
        with self.assertRaises(ValueError):
            gpu_image = mp.gpuimage(numpy_array)

    
    def test_create_invalid_gpuimage3(self):
        with self.assertRaises(ValueError):
            gpu_array = mp.gpuarray(None)


    def test_gpuarray_np_function_protocol(self):
        lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        array = np.array(lst)
        array = np.transpose(array)

        array2 = mp.gpuarray(lst)
        array2 = np.transpose(array2)
        
        npt.assert_equal(array, array2)

    
    def test_gpuimage_np_function_protocol(self):
        lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        array = np.array(lst)
        array = np.transpose(array)

        array2 = mp.gpuimage(lst)
        array2 = np.transpose(array2)
        
        npt.assert_equal(array, array2)


    def test_open_image(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie_on_gpu = mp.gpuimage(io.imread("tests/images/charlie.png"))
        charlie2 = np.array(charlie_on_gpu)
        npt.assert_almost_equal(charlie, charlie2, decimal=DECIMAL_ERROR)


    def test_rgb2grey(self):
        charlie = io.imread("tests/images/charlie.png")
        grey_charlie = rgb2gray(rgba2rgb(charlie))

        charlie_on_gpu = mp.gpuimage(io.imread("tests/images/charlie.png"))
        charlie_on_gpu.rgb2grey()
        grey_charlie2 = np.array(charlie_on_gpu)

        npt.assert_almost_equal(grey_charlie, grey_charlie2, decimal=DECIMAL_ERROR)
 

    def test_grey_and_transpose(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = np.transpose(rgb2gray(rgba2rgb(charlie)))

        charlie2 = mp.gpuimage(io.imread("tests/images/charlie.png"))
        charlie2.rgb2grey()
        charlie2.transpose()
        charlie2 = np.array(charlie2)

        npt.assert_almost_equal(charlie, charlie2, decimal=DECIMAL_ERROR)


    def test_create_invalid_operation(self):
        with self.assertRaises(ValueError):
            operation = Operation()


    def test_create_invalid_operation2(self):
        def do_nothing():
            pass

        with self.assertRaises(ValueError):
            operation = Operation(do_nothing, probability=7)


    def test_create_invalid_operation3(self):
        def do_nothing():
            pass

        with self.assertRaises(ValueError):
            operation = Operation(do_nothing, probability=1)


    def test_create_invalid_operation4(self):
        def do_nothing():
            pass

        with self.assertRaises(ValueError):
            operation = Operation(do_nothing, probability=0)


    def test_create_invalid_operation5(self):
        def do_nothing():
            pass

        with self.assertRaises(ValueError):
            operation = Operation(do_nothing, probability=-1)


    def test_create_operation(self):

        def do_nothing():
            pass

        operation = Operation(do_nothing)
        self.assertIsNotNone(operation)


    def test_create_operation2(self):

        def do_nothing():
            pass

        operation = Operation(do_nothing, probability=.6)
        self.assertIsNotNone(operation)


    def test_run_operation(self):

        def add_two_nums(x, y):
            return x + y

        operation = Operation(add_two_nums, 4, 6)
        result = operation.run()
        self.assertEqual(result, 10)


    def test_operation_grey(self):
        charlie = io.imread("tests/images/charlie.png")
        grey_charlie = rgb2gray(rgba2rgb(charlie))

        charlie_on_gpu = mp.gpuimage(io.imread("tests/images/charlie.png"))

        greyoperation = mp.Operation("rgb2grey")
        greyoperation.run_on(charlie_on_gpu)
        grey_charlie2 = np.array(charlie_on_gpu)

        npt.assert_almost_equal(grey_charlie, grey_charlie2, decimal=DECIMAL_ERROR)


    def test_operation_grey_and_transpose(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = np.transpose(rgb2gray(rgba2rgb(charlie)))

        charlie2 = mp.gpuimage(io.imread("tests/images/charlie.png"))

        operations = [mp.Operation("rgb2grey"), mp.Operation("transpose")]
        for op in operations:
            op.run_on(charlie2)

        charlie2 = np.array(charlie2)
        npt.assert_almost_equal(charlie, charlie2, decimal=DECIMAL_ERROR)

    
    def test_create_pipeline(self):
        p = mp.Pipeline([], [])
        self.assertIsNotNone(p)


    def test_create_pipeline2(self):
        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie2 = mp.gpuimage(io.imread("tests/images/charlie.png"))

        inputs = [d_charlie, d_charlie2]
        operations = [mp.Operation("rgb2grey"), mp.Operation("transpose")]

        p = mp.Pipeline(inputs, operations)
        self.assertIsNotNone(p)

    
    def test_create_pipeline3(self):
        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie2 = mp.gpuimage(io.imread("tests/images/charlie.png"))

        inputs = [d_charlie, d_charlie2]
        operations = [mp.Operation("rgb2grey"), mp.Operation("transpose")]

        p = mp.Pipeline(inputs, operations, device=0)
        self.assertIsNotNone(p)

    
    def test_create_invalid_pipeline(self):
        with self.assertRaises(ValueError):
            p = mp.Pipeline(1, [])


    def test_create_invalid_pipeline2(self):
        with self.assertRaises(ValueError):
            p = mp.Pipeline([], 1)


    def test_create_invalid_pipeline3(self):
        with self.assertRaises(ValueError):
            p = mp.Pipeline(np.array([1, 2, 3]), [])


    def test_create_invalid_pipeline4(self):
        with self.assertRaises(ValueError):
            p = mp.Pipeline([], np.array([1, 2, 3]))

    
    def test_create_invalid_pipeline5(self):
        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie2 = mp.gpuimage(io.imread("tests/images/charlie.png"))

        inputs = [d_charlie, d_charlie2]
        operations = [mp.Operation("rgb2grey"), mp.Operation("transpose")]

        with self.assertRaises(ValueError):
            p = mp.Pipeline(inputs, operations, device=5.3)


    def test_create_invalid_pipeline6(self):
        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie2 = mp.gpuimage(io.imread("tests/images/charlie.png"))

        inputs = [d_charlie, d_charlie2]
        operations = [mp.Operation("rgb2grey"), mp.Operation("transpose")]

        with self.assertRaises(ValueError):
            p = mp.Pipeline(inputs, operations, device=2, unused='test')


    def test_create_invalid_pipeline7(self):
        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie2 = mp.gpuimage(io.imread("tests/images/charlie.png"))

        inputs = [d_charlie, d_charlie2]
        operations = [mp.Operation("rgb2grey"), mp.Operation("transpose")]

        with self.assertRaises(ValueError):
            p = mp.Pipeline(inputs, operations, device="nah")
    

    def test_pipeline_run(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = np.transpose(rgb2gray(rgba2rgb(charlie)))

        charlie_on_gpu = mp.gpuimage(io.imread("tests/images/charlie.png"))
        inputs = [charlie_on_gpu]
        operations = [mp.Operation("rgb2grey"), mp.Operation("transpose")]

        p = mp.Pipeline(inputs, operations)
        p.run()

        charlie2 = np.array(charlie_on_gpu)
        npt.assert_almost_equal(charlie, charlie2, decimal=DECIMAL_ERROR)


    def test_pipeline_run2(self):
        h_charlie = io.imread("tests/images/charlie.png")
        h_charlie2 = io.imread("tests/images/charlie.png")
        h_charlie = np.transpose(rgb2gray(rgba2rgb(h_charlie)))
        h_charlie2 = np.transpose(rgb2gray(rgba2rgb(h_charlie2)))

        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie2 = mp.gpuimage(io.imread("tests/images/charlie.png"))

        inputs = [d_charlie, d_charlie2]
        operations = [mp.Operation("rgb2grey"), mp.Operation("transpose")]

        p = mp.Pipeline(inputs, operations)
        p.run()

        d_charlie = np.array(d_charlie)
        d_charlie2 = np.array(d_charlie2)

        npt.assert_almost_equal(d_charlie, h_charlie, decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(d_charlie2, h_charlie2, decimal=DECIMAL_ERROR)
    

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

        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie2 = d_charlie.clone()
        d_charlie2.rgb2grey()

        npt.assert_almost_equal(h_charlie, np.array(d_charlie),
                                decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(h_charlie2, np.array(d_charlie2),
                                decimal=DECIMAL_ERROR)


    def test_clone2(self):
        h_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        h_array2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        d_array = mp.gpuarray(h_array)
        d_array2 = d_array.clone()
        
        npt.assert_almost_equal(h_array, np.array(d_array),
                                decimal=DECIMAL_ERROR)
        npt.assert_almost_equal(h_array2, np.array(d_array2),
                                decimal=DECIMAL_ERROR)



    def test_image_from_path(self):
        h_charlie = io.imread("tests/images/charlie.png")

        d_charlie = mp.image_from_path("tests/images/charlie.png")

        npt.assert_almost_equal(h_charlie, np.array(d_charlie),
                                decimal=DECIMAL_ERROR)

    
    def test_image_from_path2(self):
        h_charlie = io.imread("tests/images/charlie.png")
        h_charlie = rgb2gray(rgba2rgb(h_charlie))

        d_charlie = mp.image_from_path("tests/images/charlie.png")
        d_charlie.rgb2grey()

        npt.assert_almost_equal(h_charlie, np.array(d_charlie),
                                decimal=DECIMAL_ERROR)

    
    def test_generator(self):
        o = mp.Operation("rgb2grey")
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


    def test_gaussian(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = rgb2gray(rgba2rgb(charlie))
        charlie = filters.gaussian(charlie, sigma=2, cval=0, truncate=8, mode="constant")
        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie.rgb2grey()
        d_charlie.gaussian(2)
        charlie2 = np.array(d_charlie)
        npt.assert_almost_equal(charlie, charlie2, decimal=4)


    def test_gamma_greyscale(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = rgb2gray(rgba2rgb(charlie))
        charlie = exposure.adjust_gamma(charlie, 2, 1)

        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie.rgb2grey()
        d_charlie.adjust_gamma(2, 1)

        npt.assert_almost_equal(charlie, np.array(d_charlie),
                                decimal=DECIMAL_ERROR)

    def test_gamma_color(self):
        charlie = io.imread("tests/images/charlie.png")
        charlie = exposure.adjust_gamma(charlie, 2, 1)

        d_charlie = mp.gpuimage(io.imread("tests/images/charlie.png"))
        d_charlie.adjust_gamma(2, 1)

        npt.assert_almost_equal(charlie, np.array(d_charlie),
                                decimal=DECIMAL_ERROR)


if __name__ == '__main__':
    unittest.main()

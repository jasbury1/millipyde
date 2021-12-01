import sys
import pathlib
cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))

import numpy as np
import numpy.testing as npt
from skimage import data, io, filters, transform, color, exposure
from skimage.io import imsave, imread
from skimage.color import rgb2gray, rgba2rgb

import timeit
import time

import millipyde as mp

def greyscale_charlie():
    print('\033[95m' + "\nGreyscaling Charlie\n" + '\033[0m')
    charlie = io.imread("examples/images/charlie.png")
    charlie_on_gpu = mp.gpuimage(charlie)

    start = time.perf_counter()
    charlie_on_gpu.rgb2grey()
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))

    start = time.perf_counter()
    imsave("output/charlie_grey.png", np.array(charlie_on_gpu))
    stop = time.perf_counter()
    print("\nTime to save: {}\n".format(stop - start))

    print('\033[95m' + "\nGreyscaling Charlie using SciKit-Image\n" + '\033[0m')
    charlie = io.imread("examples/images/charlie.png")

    start = time.perf_counter()
    grey_charlie = rgb2gray(rgba2rgb(charlie))
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))

    start = time.perf_counter()
    imsave("output/charlie_grey_skimage.png", grey_charlie)
    stop = time.perf_counter()
    print("\nTime to save: {}\n".format(stop - start))

def greyscale_and_transpose_charlie():
    print('\033[95m' + "\nGreyscaling and transposing Charlie\n" + '\033[0m')
    charlie = io.imread("examples/images/charlie.png")
    charlie_on_gpu = mp.gpuimage(charlie)

    start = time.perf_counter()
    charlie_on_gpu.rgb2grey()
    charlie_on_gpu.transpose()
    stop = time.perf_counter()

    print("\nTime to convert image: {}\n".format(stop - start))

    print('\033[95m' + "\nGreyscaling and transposing Charlie using SciKit-Image\n" + '\033[0m')
    charlie = io.imread("examples/images/charlie.png")

    start = time.perf_counter()
    grey_charlie = rgb2gray(rgb2gray(rgba2rgb(charlie)))
    transform.rotate(grey_charlie, 90)
    stop = time.perf_counter()

    print("\nTime to convert image: {}\n".format(stop - start))


def greyscale_and_transpose_pipeline():
    print('\033[95m' + "\nGreyscaling and transposing Charlie without pipeline\n" + '\033[0m')
    d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))
    d_charlie2 = mp.gpuimage(io.imread("examples/images/charlie.png"))
    start = time.perf_counter()
    d_charlie.rgb2grey()
    d_charlie.transpose()
    d_charlie2.rgb2grey()
    d_charlie2.transpose()
    stop = time.perf_counter()
    print("\nTime to beat: {}\n".format(stop - start))

    print('\033[95m' + "\nGreyscaling and transposing Charlie with pipeline\n" + '\033[0m')
    d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))
    d_charlie2 = mp.gpuimage(io.imread("examples/images/charlie.png"))
    inputs = [d_charlie, d_charlie2]
    operations = [mp.Operation("rgb2grey"), mp.Operation("transpose")]
    p = mp.Pipeline(inputs, operations)
    start = time.perf_counter()
    p.run()
    stop = time.perf_counter()
    print("\nTime to run pipeline: {}\n".format(stop - start))


def greyscale_and_transpose_pipeline2():
    
    print('\033[95m' + "\nGreyscaling and transposing Charlie without pipeline\n" + '\033[0m')
    d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))
    d_charlie2 = mp.gpuimage(io.imread("examples/images/charlie.png"))
    d_charlie3 = mp.gpuimage(io.imread("examples/images/charlie.png"))
    d_charlie4 = mp.gpuimage(io.imread("examples/images/charlie.png"))
    start = time.perf_counter()
    d_charlie.rgb2grey()
    d_charlie.transpose()
    d_charlie.transpose()
    d_charlie2.rgb2grey()
    d_charlie2.transpose()
    d_charlie2.transpose()
    d_charlie3.rgb2grey()
    d_charlie3.transpose()
    d_charlie3.transpose()
    d_charlie4.rgb2grey()
    d_charlie4.transpose()
    d_charlie4.transpose()
    stop = time.perf_counter()
    print("\nTime to beat: {}\n".format(stop - start))

    print('\033[95m' + "\nGreyscaling and transposing Charlie with pipeline\n" + '\033[0m')
    d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))
    d_charlie2 = mp.gpuimage(io.imread("examples/images/charlie.png"))
    d_charlie3 = mp.gpuimage(io.imread("examples/images/charlie.png"))
    d_charlie4 = mp.gpuimage(io.imread("examples/images/charlie.png"))
    #inputs = [d_charlie, d_charlie2, d_charlie3, d_charlie4]
    inputs = [d_charlie, d_charlie2, d_charlie3, d_charlie4]
    operations = [mp.Operation("rgb2grey"), mp.Operation("transpose"), mp.Operation("transpose")]
    p = mp.Pipeline(inputs, operations)
    start = time.perf_counter()
    p.run()
    stop = time.perf_counter()
    print("\nTime to run pipeline: {}\n".format(stop - start))


def gaussian_charlie():
    d_charlie = mp.gpuimage(io.imread("examples/images/happyboy.png"))
    d_charlie.rgb2grey()
    d_charlie.transpose()
    #d_charlie.gaussian()
    print("saving")
    imsave("gaussian.png", np.array(d_charlie))


def greyscale_performance():
    print('\033[95m' + "\nGreyscaling Charlie Small with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieSmall.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.rgb2grey()
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\nGreyscaling Charlie Small using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = rgb2gray(rgba2rgb(charlie))
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))

    print('\033[95m' + "\nGreyscaling Charlie Medium with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieMedium.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.rgb2grey()
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\nGreyscaling Charlie Medium using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = rgb2gray(rgba2rgb(charlie))
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))

    print('\033[95m' + "\nGreyscaling Charlie Large with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieLarge.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.rgb2grey()
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\nGreyscaling Charlie Large using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = rgb2gray(rgba2rgb(charlie))
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


    print('\033[95m' + "\nGreyscaling Charlie Extra Large with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieExtraLarge.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.rgb2grey()
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\nGreyscaling Charlie Extra Large using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = rgb2gray(rgba2rgb(charlie))
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))

def transpose_performance():
    print('\033[95m' + "\Transposing Charlie Small with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieSmall.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.transpose()
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\Transposing Charlie Small using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = np.transpose(charlie)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))

    print('\033[95m' + "\Transposing Charlie Medium with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieMedium.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.transpose()
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\Transposing Charlie Medium using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = np.transpose(charlie)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))

    print('\033[95m' + "\Transposing Charlie Large with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieLarge.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.transpose()
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\Transposing Charlie Large using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = np.transpose(charlie)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


    print('\033[95m' + "\Transposing Charlie Extra Large with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieExtraLarge.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.transpose()
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\Transposing Charlie Extra Large using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = np.transpose(charlie)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


def gauss_performance():
    charlie = io.imread("examples/benchmark_in/charlie12.png")
    charlie = rgb2gray(charlie)
    start = time.perf_counter()
    charlie = filters.gaussian(charlie, sigma=2, cval=0, truncate=8, mode="constant")
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))

    d_charlie = mp.gpuimage(io.imread("examples/benchmark_in/charlie12.png"))
    d_charlie.rgb2grey()
    start = time.perf_counter()
    d_charlie.gaussian(2)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


def rot_performance():
    '''
    charlie = io.imread("examples/benchmark_in/charlieMedium.png")
    charlie = rgb2gray(charlie)
    charlie = filters.gaussian(charlie, sigma=5, truncate=8)
    d_charlie = mp.gpuimage(io.imread("examples/benchmark_in/charlieMedium.png"))
    d_charlie.rgb2grey()
    d_charlie.gaussian(2)
    imsave("gauss2.png", np.array(d_charlie))
    '''


    print('\033[95m' + "\Transposing Charlie Small with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieSmall.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.rotate(.785398)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\Transposing Charlie Small using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = transform.rotate(charlie, 45)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))

    print('\033[95m' + "\Transposing Charlie Medium with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieMedium.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.rotate(.785398)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\Transposing Charlie Medium using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = transform.rotate(charlie, 45)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))

    print('\033[95m' + "\Transposing Charlie Large with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieLarge.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.rotate(.785398)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\Transposing Charlie Large using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = transform.rotate(charlie, 45)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


    print('\033[95m' + "\Transposing Charlie Extra Large with {} iterations\n" + '\033[0m')
    charlie = io.imread("examples/benchmark_in/charlieExtraLarge.png")
    d_charlie = mp.gpuimage(charlie)

    start = time.perf_counter()
    d_charlie.rotate(.785398)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start)) 

    print('\033[95m' + "\Transposing Charlie Extra Large using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    charlie = transform.rotate(charlie, 45)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


def gamma_performance():
    charlie = io.imread("examples/benchmark_in/charlie6.png")
    charlie = color.rgb2grey(charlie)

    start = time.perf_counter()
    charlie = exposure.adjust_gamma(charlie, 2, 1)
    stop = time.perf_counter()
    print("\nSKImage Time: {}\n".format(stop - start))

    d_charlie = mp.gpuimage(io.imread("examples/benchmark_in/charlie6.png"))
    d_charlie.rgb2grey()

    start = time.perf_counter()
    d_charlie.adjust_gamma(2, 1)
    stop = time.perf_counter()
    print("\nMillipyde Time: {}\n".format(stop - start))

    npt.assert_almost_equal(charlie, np.array(d_charlie), decimal=4)

def pipeline_performance():
    num_inputs = 100

    ops = [mp.Operation("gaussian", 2),
           mp.Operation("adjust_gamma", 2, 1),
           mp.Operation("fliplr"),
           mp.Operation("rotate", 45),
           mp.Operation("rgb2grey")
           ]
    
    inputs = []
    for i in range(num_inputs):
        inputs.append(mp.image_from_path("examples/benchmark_in/charlie6.png"))

    pipeline = mp.Pipeline(inputs, ops, device=0)

    start = time.perf_counter()
    pipeline.run()
    stop = time.perf_counter()
    print("\nTime to run: {}\n".format(stop - start))
    
    inputs = []
    for i in range(num_inputs):
        inputs.append(mp.image_from_path("examples/benchmark_in/charlie6.png"))

    start = time.perf_counter()
    for i in inputs:
        for o in ops:
            o.run_on(i)
    stop = time.perf_counter()
    print("\nTime to run non-pipeline version: {}\n".format(stop - start))


def long_pipeline_performance():
    num_inputs = 100

    ops = [mp.Operation("gaussian", 2),
           mp.Operation("rotate", 4),
           mp.Operation("adjust_gamma", 1.3, 1),
           mp.Operation("rotate", 10),
           mp.Operation("gaussian", 2),
           mp.Operation("adjust_gamma", 1.4, 1),
           mp.Operation("fliplr"),
           mp.Operation("adjust_gamma", .5, 1),
           mp.Operation("fliplr"),
           mp.Operation("gaussian", 2),
           mp.Operation("rotate", 2),
           mp.Operation("gaussian", 2),
           mp.Operation("gaussian", 2),
           mp.Operation("rotate", 10),
           mp.Operation("rgb2grey"),
           mp.Operation("gaussian", 2),
           mp.Operation("gaussian", 2),
           mp.Operation("rotate", 4),
           mp.Operation("adjust_gamma", 1.3, 1),
           mp.Operation("rotate", 10)
           ]

    inputs = []

    for i in range(num_inputs):
        inputs.append(mp.image_from_path("examples/benchmark_in/charlie6.png"))

    pipeline = mp.Pipeline(inputs, ops, device=0)

    start = time.perf_counter()
    pipeline.run()
    stop = time.perf_counter()
    print("\nTime to run: {}\n".format(stop - start))
    
    inputs = []
    for i in range(num_inputs):
        inputs.append(mp.image_from_path("examples/benchmark_in/charlie6.png"))

    start = time.perf_counter()
    for i in inputs:
        for o in ops:
            o.run_on(i)
    stop = time.perf_counter()
    print("\nTime to run non-pipeline version: {}\n".format(stop - start))


'''
struct timespec ts;
    ts.tv_sec = 2/1000;
    ts.tv_nsec = (2 % 1000) * 1000000;

    nanosleep(&ts, &ts);
'''


def connected_pipelines_performance():
    num_inputs = 75

    ops = [mp.Operation("gaussian", 2),
           mp.Operation("rotate", 4),
           mp.Operation("adjust_gamma", 1.3, 1),
           mp.Operation("rotate", 10),
           mp.Operation("gaussian", 2),
           mp.Operation("adjust_gamma", 1.4, 1),
           mp.Operation("fliplr"),
           mp.Operation("adjust_gamma", .5, 1),
           mp.Operation("fliplr"),
           mp.Operation("gaussian", 2),
           ]

    ops2 = [mp.Operation("rotate", 2),
            mp.Operation("gaussian", 2),
            mp.Operation("gaussian", 2),
            mp.Operation("rotate", 10),
            mp.Operation("rgb2grey"),
            mp.Operation("gaussian", 2),
            mp.Operation("gaussian", 2),
            mp.Operation("rotate", 4),
            mp.Operation("adjust_gamma", 1.3, 1),
            mp.Operation("rotate", 10)
            ]

    inputs = []
    for i in range(num_inputs):
        inputs.append(mp.image_from_path("examples/benchmark_in/charlie6.png"))

    start = time.perf_counter()
    for i in inputs:
        for o in ops:
            o.run_on(i)
        for o in ops2:
            o.run_on(i)
    stop = time.perf_counter()
    print("\nControl: {}\n".format(stop - start))

    inputs = []

    for i in range(num_inputs):
        inputs.append(mp.image_from_path("examples/benchmark_in/charlie6.png"))

    pipeline = mp.Pipeline(inputs, ops, device=0)
    pipeline2 = mp.Pipeline([], ops2, device=1)
    pipeline.connect_to(pipeline2)

    start = time.perf_counter()
    pipeline.run()
    stop = time.perf_counter()
    print("\nTime to run connected pipeline: {}\n".format(stop - start))


    ops3 = [mp.Operation("gaussian", 2),
           mp.Operation("rotate", 4),
           mp.Operation("adjust_gamma", 1.3, 1),
           mp.Operation("rotate", 10),
           mp.Operation("gaussian", 2),
           mp.Operation("adjust_gamma", 1.4, 1),
           mp.Operation("fliplr"),
           mp.Operation("adjust_gamma", .5, 1),
           mp.Operation("fliplr"),
           mp.Operation("gaussian", 2),
           mp.Operation("rotate", 2),
           mp.Operation("gaussian", 2),
           mp.Operation("gaussian", 2),
           mp.Operation("rotate", 10),
           mp.Operation("rgb2grey"),
           mp.Operation("gaussian", 2),
           mp.Operation("gaussian", 2),
           mp.Operation("rotate", 4),
           mp.Operation("adjust_gamma", 1.3, 1),
           mp.Operation("rotate", 10)
           ]

    inputs = []

    for i in range(num_inputs):
        inputs.append(mp.image_from_path("examples/benchmark_in/charlie6.png"))

    pipeline = mp.Pipeline(inputs, ops3)

    start = time.perf_counter()
    pipeline.run()
    stop = time.perf_counter()
    print("\nTime to run one pipeline: {}\n".format(stop - start))


def connected_pipelines_performance_2():
    num_inputs = 75

    ops = [mp.Operation("gaussian", 2),
           mp.Operation("adjust_gamma", 2, 1),
           mp.Operation("fliplr")
           ]

    ops2 = [mp.Operation("rotate", 45),
            mp.Operation("rgb2grey")
            ]
    
    inputs = []
    for i in range(num_inputs):
        inputs.append(mp.image_from_path("examples/benchmark_in/charlie6.png"))

    pipeline = mp.Pipeline(inputs, ops, device=0)
    pipeline2 = mp.Pipeline([], ops2, device=1)

    pipeline.connect_to(pipeline2)

    start = time.perf_counter()
    pipeline.run()
    stop = time.perf_counter()
    print("\nTime to run: {}\n".format(stop - start))


def main():
    num_inputs = 75

    ops = [mp.Operation("gaussian", 2),
           mp.Operation("rotate", 4),
           mp.Operation("adjust_gamma", 1.3, 1),
           mp.Operation("rotate", 10),
           mp.Operation("gaussian", 2),
           mp.Operation("adjust_gamma", 1.4, 1),
           mp.Operation("fliplr"),
           mp.Operation("adjust_gamma", .5, 1),
           mp.Operation("fliplr"),
           mp.Operation("gaussian", 2),
           mp.Operation("rotate", 2),
           mp.Operation("gaussian", 2),
           mp.Operation("gaussian", 2),
           mp.Operation("rotate", 10),
           mp.Operation("rgb2grey"),
           mp.Operation("gaussian", 2),
           mp.Operation("gaussian", 2),
           mp.Operation("rotate", 4),
           mp.Operation("adjust_gamma", 1.3, 1),
           mp.Operation("rotate", 10)
           ]

    inputs = []
    for _ in range(num_inputs):
        inputs.append(mp.image_from_path("examples/benchmark_in/charlie6.png"))

    pipeline = mp.Pipeline(inputs, ops)

    start = time.perf_counter()
    pipeline.run()
    stop = time.perf_counter()
    print("\nTime to run one pipeline: {}\n".format(stop - start))


    control = np.array(inputs[0])
    for i in range(1, num_inputs):
        print(i)
        experiment = np.array(inputs[i])
        npt.assert_almost_equal(control, experiment, decimal=4)

    
    


if __name__ == '__main__':
    main()

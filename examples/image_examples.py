import sys
import pathlib
cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))

import numpy as np
from skimage import data, io, filters, transform
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
    d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))
    d_charlie.rgb2grey()
    d_charlie.transpose()
    #d_charlie.gaussian()
    print("saving")
    imsave("gaussian.png", np.array(d_charlie))





def main():
    #greyscale_charlie()
    #greyscale_and_transpose_charlie()
    #greyscale_and_transpose_pipeline()
    #greyscale_and_transpose_pipeline2()
    #gaussian_charlie()
    d_charlie = mp.gpuimage(io.imread("examples/images/charlie.png"))
    d_charlie.rgb2grey()
    d_charlie.transpose()
    #d_charlie.rgb2grey()
    #d_charlie.rotate()
    imsave("testboi.png", np.array(d_charlie))




if __name__ == '__main__':
    main()
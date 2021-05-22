import numpy as np
from skimage import data, io, filters
from skimage.io import imsave, imread
from skimage.color import rgb2gray, rgba2rgb

import timeit
import time

import millipyde as mp

def greyscale_charlie():
    print("Greyscaling Charlie")
    charlie = io.imread("examples/images/charlie.png")
    charlie_on_gpu = mp.gpuarray(charlie)

    start = time.perf_counter()
    charlie_on_gpu.rgb2grey()
    stop = time.perf_counter()

    print("\nTime to convert image: {}\n".format(stop - start))

    start = time.perf_counter()
    imsave("output/charlie_grey.png", charlie_on_gpu.__array__())
    stop = time.perf_counter()
    print("\nTime to save: {}\n".format(stop - start))

def main():
    greyscale_charlie()


if __name__ == '__main__':
    main()
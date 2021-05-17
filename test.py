import numpy as np
from skimage import data, io, filters
from skimage.io import imsave, imread
from skimage.color import rgb2gray, rgba2rgb

import timeit
import time

import millipyde as mp

mp.test_func()
'''
nparr = np.array([1, 2, 3, 4])
arr = mp.GPUArray(nparr)

sins = 10 * np.sin(arr)
print(sins)

arr.add_one()

print(arr.__array__())

sins = 10 * np.sin(arr)

print(sins)

image = data.chelsea()

print(image)

gpuimg = mp.GPUArray(image)

imsave("test.png", image)

'''

charlie = io.imread("examples/images/charlie.png")
charlie_on_gpu = mp.GPUArray(charlie)

start = time.perf_counter()
charlie_on_gpu.to_greyscale()
stop = time.perf_counter()
print("\nTime to calculate: {}\n".format(stop - start))

start = time.perf_counter()
imsave("output/charlie_grey.png", charlie_on_gpu.__array__())
stop = time.perf_counter()
print("\nTime to save: {}\n".format(stop - start))

start = time.perf_counter()
grey_charlie = rgb2gray(rgb2gray(rgba2rgb(charlie)))
stop = time.perf_counter()
print("\nSKImage time to calculate: {}\n".format(stop - start))

start = time.perf_counter()
imsave("output/charlie_grey_skimage.png", grey_charlie)
stop = time.perf_counter()
print("\nSKImage time to save: {}\n".format(stop - start))







print("Done")
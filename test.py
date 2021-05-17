import numpy as np
from skimage import data, io, filters
from skimage.io import imsave, imread
from skimage.color import rgb2gray
import timeit

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
img = data.chelsea()
gpuimg = mp.GPUArray(img)
gpuimg.to_greyscale()
imsave("chelsea_original.png", img)
imsave("chelsea_grey.png", gpuimg.__array__())



charlie = io.imread("examples/images/charlie.png")
charlie_on_gpu = mp.GPUArray(charlie)
charlie_on_gpu.to_greyscale()
imsave("output/charlie_original.png", charlie)
imsave("output/charlie_grey.png", charlie_on_gpu.__array__())






print("Done")
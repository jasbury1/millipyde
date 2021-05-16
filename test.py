import numpy as np
from skimage import data, io, filters
from skimage.io import imsave, imread

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
nparr = np.array([[1, 2], [3, 4], [5, 6]])
gpuarr = mp.GPUArray(nparr)
result = gpuarr.to_greyscale()
print(result)

'''
image = data.chelsea()
gpuarr2 = mp.GPUArray(image)
gpuarr2.to_greyscale()
'''





print("Done")
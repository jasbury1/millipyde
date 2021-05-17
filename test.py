import numpy as np
from skimage import data, io, filters
from skimage.io import imsave, imread
from skimage.color import rgb2gray

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
print(img)
gpuimg = mp.GPUArray(img)

gpuimg.to_greyscale()
result_img = gpuimg.__array__()
print(result_img[0][0])
print(img[0][0])
imsave("tests/test2.png", gpuimg.__array__())
'''
image = data.chelsea()
gpuarr2 = mp.GPUArray(image)
gpuarr2.to_greyscale()
'''





print("Done")
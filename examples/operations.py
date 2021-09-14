import sys
import pathlib
cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))

import numpy as np
from skimage import data, io, filters, transform
from skimage.io import imsave, imread

import timeit
import time

import millipyde as mp

def say_hello(num):
    print("Hello world")
    print(num)

def say_hi():
    print("Hi")

def main():
    '''
    operation = mp.Operation(say_hello, 5)
    operation2 = mp.Operation(say_hi)

    operation.run()
    operation2.run()

    charlie = io.imread("examples/images/charlie.png")
    charlie_on_gpu = mp.gpuimage(charlie)
    charlie_on_gpu.rgb2grey()
    imsave("output/operations.png", np.array(charlie_on_gpu))
    charlie = io.imread("examples/images/charlie.png")
    charlie_on_gpu = mp.gpuimage(charlie)
     

    mp.test_func3(charlie_on_gpu, "rgb2grey")
    '''

    arr = mp.gpuarray(np.array([1, 2, 3, 4]))
    operation = mp.Operation("print_two_ints", 4, 5)

    operation.run_on(arr)

    operation2 = mp.Operation(say_hi)
    operation2.run()
    


if __name__ == '__main__':
    main()

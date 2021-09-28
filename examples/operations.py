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
    charlie_on_gpu = mp.gpuimage(io.imread("examples/images/charlie.png"))
    inputs = [charlie_on_gpu]
    operations = [mp.Operation("rgb2grey"), mp.Operation("transpose")]

    p = mp.Pipeline(inputs, operations)
    p.start()
    


if __name__ == '__main__':
    main()

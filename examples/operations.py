import sys
import pathlib
cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))

import numpy as np

import timeit
import time

import millipyde as mp

def say_hello(num):
    print("Hello world")
    print(num)

def say_hi():
    print("Hi")

def main():
    operation = mp.Operation(say_hello, 5)
    operation2 = mp.Operation(say_hi)
    operation3 = mp.Operation(mp.test_func)


if __name__ == '__main__':
    main()

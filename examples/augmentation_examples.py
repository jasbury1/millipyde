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

def augment():
    aug_seq = mp.augmentation_seq([

    ])

    images = ...
    



def main():
    augment()


if __name__ == '__main__':
    main()

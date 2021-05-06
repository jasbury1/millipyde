import numpy as np

import sys
sys.path.insert(1, '../')
#from millipyde import *

import millipyde as mp

#array = mp.GPUArray()

#a = np.arange(15)
#test_func(a)

#print(type(a))

#array = mp.GPUArray()

#mp.test_func(4)

#arr = mp.GPUArray()

nparr = np.array([1, 2, 3, 4])
arr = mp.GPUArray(nparr)
sins = 10 * np.sin(arr)
print(sins)

arr.add_one()

print("Done")
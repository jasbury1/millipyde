import numpy as np

import sys
sys.path.insert(1, '../')
#from millipyde import *

import millipyde as mp

mp.test_func()

nparr = np.array([1, 2, 3, 4])
arr = mp.GPUArray(nparr)
sins = 10 * np.sin(arr)
print(sins)

arr.add_one()

print("Done")
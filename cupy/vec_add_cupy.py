#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import cupy as cp
import numpy as np

N = 100000 # length of the vectors

# create vectors of random numbers and transfer them to the GPU:
x = cp.asarray( np.random.rand( N ) )
y = cp.asarray( np.random.rand( N ) )

# perform the calculations on the GPU:
z = x + y

# transfer the result back from the GPU
z_cmp = z.asnumpy( z )

# verify that the result is correct
diff = np.fabs( z_cmp - ( cp.asnumpy( x ) + cp.asnumpy( y ) ) )

print( "|diff| = {0:.16f}".format( np.linalg.norm( diff, ) ) )
# |diff| = 0.0000000000000000

#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

# create vectors with random numbers on the CPU
N = np.int32( 100000 )
x_cpu = np.random.rand( N ).astype( np.float64 )
y_cpu = np.random.rand( N ).astype( np.float64 )

# create the OpenCL context and a queue
ctx = cl.create_some_context()
queue = cl.CommandQueue( ctx )

# allocate memory on the device memory AND transfer x and y data to the device!
mem_flags = cl.mem_flags

x = cl.Buffer( ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=x_cpu )
y = cl.Buffer( ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=y_cpu )
z = cl.Buffer( ctx, mem_flags.WRITE_ONLY, x_cpu.nbytes )

# prepare the kernel:
program_src = \
    """
    __kernel void add_vec_kernel(
	    __global double const* restrict x,
	    __global double const* restrict y,
	    __global double* restrict z, int const n )
	{
	    int const gid = get_global_id( 0 );

	    if( gid < n ) z[ gid ] = x[ gid ] + y[ gid ];
	}
	"""

# compile the program containing the kernel
prg = cl.Program( ctx, program_src ).build()

# execute the kernel using the prepared arguments
prg.add_vec_kernel( queue, x_cpu.shape, None, x, y, z, N )

# prepare space for the result vector on the CPU and transfer the memory back
z_cpu = np.empty_like( x_cpu )
cl.enqueue_copy( queue, z_cpu, z )

# verify that the calculation was correct:
diff = np.fabs( z_cpu - ( x_cpu + y_cpu ) )

print( "|diff| = {0:.16f}".format( np.linalg.norm( diff, ) ) )
# |diff| = 0.0000000000000000


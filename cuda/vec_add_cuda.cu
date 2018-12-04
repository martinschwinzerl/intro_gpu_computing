#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda.h>

/* Define the kernel function: */

__global__ void add_vec_kernel(
    double const* __restrict__ x, double const* __restrict__ y,
    double* __restrict__ z, int const n )
{
    /* blockIdx, blockDim and threadIdx are variables describing the
     * dimensions of the "grid" which are automatically provided by the
     * Cuda runtime */

    int const gid = blockIdx.x * blockDim.x + threadIdx.x;

    if( gid < n )
    {
        z[ gid ] = x[ gid ] + y[ gid ];
    }

    return;
}

int main( void )
{
    /* ---------------------------------------------------------------------- */
    /* prepare the host vectors: */

    int32_t const N = int32_t{ 10000 };

    std::vector< double > x( N, double{ 0.0 } );
    std::vector< double > y( N, double{ 0.0 } );
    std::vector< double > z( N, double{ 0.0 } );

    std::mt19937_64 prng( 20181205u );
    std::uniform_real_distribution< double >
        dist( double{ -10. }, double{ +10. } );

    for( int32_t ii = int32_t{ 0 } ; ii < N ; ++ii )
    {
        x[ ii ] = dist( prng );
        y[ ii ] = dist( prng );
    }

    cudaError_t cu_err;

    /* --------------------------------------------------------------------- */
    /* use the "default" / "first" Cuda device for the program: */

    int device = int{ 0 };
    ::cudaGetDevice( &device );
    cu_err = ::cudaDeviceSynchronize();
    assert( cu_err == ::cudaSuccess );

    /* --------------------------------------------------------------------- */
    /* Allocate the buffers on the device */
    /* x_arg, y_arg, z_arg ... handles on the host side managing buffers in *
     * the device memory */

    double* x_arg = nullptr;
    double* y_arg = nullptr;
    double* z_arg = nullptr;

    ::cudaMalloc( &x_arg, sizeof( double ) * N );
    ::cudaMalloc( &y_arg, sizeof( double ) * N );
    ::cudaMalloc( &z_arg, sizeof( double ) * N );

    /* --------------------------------------------------------------------- */
    /* Transfer x and y from host to device */

    ::cudaMemcpy( x_arg, x.data(), sizeof( double ) * N, cudaMemcpyHostToDevice );
    ::cudaMemcpy( y_arg, y.data(), sizeof( double ) * N, cudaMemcpyHostToDevice );

    /* --------------------------------------------------------------------- */
    /* execute kernel on the device */

    int32_t const threads_per_block = int32_t{ 128 };

    int32_t const num_blocks =
        ( N + threads_per_block - int32_t{ 1 } ) / threads_per_block;

    add_vec_kernel<<< num_blocks, threads_per_block >>>( x_arg, y_arg, z_arg, N );

    cu_err = ::cudaPeekAtLastError();
    assert( cu_err == ::cudaSuccess );

    /* -------------------------------------------------------------------- */
    /* transfer the result from the device buffer to the host buffer */

    ::cudaMemcpy( z.data(), z_arg, sizeof( double ) * N, cudaMemcpyDeviceToHost );

    /* ------------------------------------------------------------------- */
    /* verify that the result is correct */

    bool success = true;
    double const EPS = std::numeric_limits< double >::epsilon();

    for( int32_t ii = int32_t{ 0 } ; ii < N ; ++ii )
    {
        if( std::fabs( ( x[ ii ] + y[ ii ] ) - z[ ii ] ) > EPS )
        {
            success = false;
            break;
        }
    }

    std::cout << "Success: " << std::boolalpha << success << std::endl;

    /* -------------------------------------------------------------------- */
    /* Clean-up */

    ::cudaFree( x_arg );
    ::cudaFree( y_arg );
    ::cudaFree( z_arg );

    return 0;
}

/* end: */

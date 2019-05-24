#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#if defined( __GNUC__ ) && __GNUC__ >= 6 
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif 

#include <CL/cl2.hpp>

#if defined( __GNUC__ ) && __GNUC__ >= 6
    #pragma GCC diagnostic pop
#endif

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

    /* --------------------------------------------------------------------- */
    /* Select the first device on the first platform: */
    std::vector< cl::Platform > platforms;
    cl::Platform::get( &platforms );

    assert( !platforms.empty() );
    std::vector< cl::Device > devices;

    bool device_found = false;
    cl::Device device;

    for( auto const& available_platform : platforms )
    {
        devices.clear();
        available_platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );

        if( !devices.empty() )
        {
            device = devices.front();
            device_found = true;
            break;
        }
    }

    assert( device_found );

    /* --------------------------------------------------------------------- */
    /* Build the program and get the kernel: */

    cl::Context context( device );
    cl::CommandQueue queue( context, device );

    std::string const kernel_source =
        "__kernel void add_vec_kernel( \r\n"
             " __global double const* restrict x,"
             " __global double const* restrict y,"
             " __global double* restrict z,"
             " int const n )\r\n"
        "{\r\n"
        "   int const gid = get_global_id( 0 );\r\n"
        "   if( gid < n ) \r\n"
        "   {\r\n"
        "       z[ gid ] = x[ gid ] + y[ gid ]; \r\n"
        "   }\r\n"
        "}\r\n";

    std::string const compile_options = "-w -Werror";

    cl::Program program( context, kernel_source );
    cl_int ret = program.build( compile_options.c_str() );

    assert( ret == CL_SUCCESS );
    cl::Kernel kernel( program, "add_vec_kernel" );

    /* --------------------------------------------------------------------- */
    /* Allocate the buffers on the device */
    /* x_arg, y_arg, z_arg ... handles on the host side managing buffers in *
     * the device memory */

    cl::Buffer x_arg( context, CL_MEM_READ_WRITE, sizeof( double ) * N, nullptr );
    cl::Buffer y_arg( context, CL_MEM_READ_WRITE, sizeof( double ) * N, nullptr );
    cl::Buffer z_arg( context, CL_MEM_READ_WRITE, sizeof( double ) * N, nullptr );

    /* Transfer x and y from the host to the device */

    ret = queue.enqueueWriteBuffer( x_arg, CL_TRUE, std::size_t{ 0 },
                                    x.size() * sizeof( double ), x.data() );

    assert( ret == CL_SUCCESS );

    ret = queue.enqueueWriteBuffer( y_arg, CL_TRUE, std::size_t{ 0 },
                                    y.size() * sizeof( double ), y.data() );

    assert( ret == CL_SUCCESS );

    /* --------------------------------------------------------------------- */
    /* Prepare the kernel for execution: bind the arguments to the kernel */

    kernel.setArg( 0, x_arg );
    kernel.setArg( 1, y_arg );
    kernel.setArg( 2, z_arg );
    kernel.setArg( 3, N );

    /* -------------------------------------------------------------------- */
    /* execute the kernel on the device */
    cl::NDRange offset = cl::NullRange;
    cl::NDRange local  = cl::NullRange;

    ret = queue.enqueueNDRangeKernel( kernel, offset, N, local );
    assert( ret == CL_SUCCESS );

    /* -------------------------------------------------------------------- */
    /* transfer the result from the device buffer to the host buffer */

    ret = queue.enqueueReadBuffer( z_arg, CL_TRUE, std::size_t{ 0 },
                                   z.size() * sizeof( double ), z.data() );

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

    return 0;
}

/* end: opencl/vec_add_opencl.cpp */

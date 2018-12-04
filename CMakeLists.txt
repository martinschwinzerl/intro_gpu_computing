cmake_minimum_required( VERSION 3.8 )
project( intro_gpu_computing )

option( INTRO_ENABLE_OPENCL "Enable OpenCL" ON  )
option( INTRO_ENABLE_CUDA   "Enable Cuda"   ON )

enable_language( C )
enable_language( CXX )

if( INTRO_ENABLE_CUDA )
    enable_language( CUDA )
endif()

#  ----------------------------------------------------------------------

if( INTRO_ENABLE_OPENCL )
    add_subdirectory( opencl )
endif()


if( INTRO_ENABLE_CUDA )
    add_subdirectory( cuda )
endif()


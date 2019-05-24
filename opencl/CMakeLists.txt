find_package( OpenCL REQUIRED )

# -----------------------------------------------------------------------------
# Example program vec_add_opencl_cxx:
# vector vector addition, writting using the C++ Host bindings

add_executable( vec_add_opencl_cxx vec_add_opencl.cpp )

target_link_libraries( vec_add_opencl_cxx ${OpenCL_LIBRARIES} )
target_include_directories( vec_add_opencl_cxx PRIVATE ${OpenCL_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} )

target_compile_definitions( vec_add_opencl_cxx PRIVATE
    -DCL_HPP_MINIMUM_OPENCL_VERSION=110
    -DCL_HPP_TARGET_OPENCL_VERSION=120
    -DHAVE_CL2  )

set_property( TARGET vec_add_opencl_cxx PROPERTY CXX_STANDARD 11 )
set_property( TARGET vec_add_opencl_cxx PROPERTY CXX_STANDARD_REQUIRED ON )


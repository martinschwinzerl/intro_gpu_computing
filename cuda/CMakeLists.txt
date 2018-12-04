add_executable( vec_add_cuda_cxx vec_add_cuda.cu )

set_target_properties( vec_add_cuda_cxx PROPERTIES CXX_STANDARD 11 )
set_target_properties( vec_add_cuda_cxx PROPERTIES CXX_STANDARD_REQUIRED ON )
set_target_properties( vec_add_cuda_cxx PROPERTIES CUDA_STANDARD 11 )
set_target_properties( vec_add_cuda_cxx PROPERTIES CUDA_STANDARD_REQUIRED 11 )

target_compile_definitions( vec_add_cuda_cxx PRIVATE
    -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES  )

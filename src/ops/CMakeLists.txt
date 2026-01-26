add_library(mori_ops dispatch_combine/dispatch_combine.cpp
                     dispatch_combine/internode_v1.cpp
                     dispatch_combine/layout_transform_kernels.cpp)
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/)
target_link_libraries(mori_ops mori_application mori_shmem mori_logging
                      hip::host hip::device)
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(ENABLE_PROFILER)
  target_link_libraries(mori_ops mori_profiler_interface)
endif()
# DeepEP-isolated dispatch/combine implementation
add_library(mori_ops_deepep dispatch_combine_deepep/dispatch_combine_deepep.cpp)
target_include_directories(mori_ops_deepep PUBLIC ${CMAKE_SOURCE_DIR}/)
target_link_libraries(mori_ops_deepep mori_application mori_shmem mori_logging
                      hip::host hip::device)
target_include_directories(mori_ops_deepep PUBLIC ${CMAKE_SOURCE_DIR}/include)

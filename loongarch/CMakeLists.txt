cmake_minimum_required(VERSION 3.5)
project(run_model)

# Include directories
include_directories("/home/jostar/workspace/torch/torch/include")
include_directories("/home/jostar/workspace/torch/torch/include/torch/csrc/api/include")

# Set C++ compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wcast-align -mstrict-align")

# Set CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "/home/jostar/workspace/torch/torch")

# Check if environment variable is set for the library path
if(DEFINED ENV{LIBRARY_PATH})
    set(LIBRARY_PATH $ENV{LIBRARY_PATH})
else()
    set(LIBRARY_PATH "/home/jostar/workspace/torch/torch/lib")
endif()

# Add executable
add_executable(run_model run_model.cpp)

# Link libraries using the environment variable
target_link_libraries(run_model "${LIBRARY_PATH}/libtorch.so")
target_link_libraries(run_model "${LIBRARY_PATH}/libtorch_cpu.so")
target_link_libraries(run_model "${LIBRARY_PATH}/libc10.so")
target_link_libraries(run_model "${LIBRARY_PATH}/libopenblas.so.0")
target_link_libraries(run_model "${LIBRARY_PATH}/libbackend_with_compiler.so")
target_link_libraries(run_model "${LIBRARY_PATH}/libshm.so")
target_link_libraries(run_model "${LIBRARY_PATH}/libtorch_global_deps.so")
target_link_libraries(run_model "${LIBRARY_PATH}/libtorchbind_test.so")
target_link_libraries(run_model "${LIBRARY_PATH}/libjitbackend_test.so")

set_property(TARGET run_model PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${LIBRARY_PATH}/*.dll")
  add_custom_command(TARGET run_model
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:run_model>)
endif (MSVC)
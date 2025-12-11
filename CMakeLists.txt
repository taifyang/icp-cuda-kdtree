cmake_minimum_required(VERSION 3.18)
set(CMAKE_BUILD_TYPE "Release")
project(icp_demo LANGUAGES C CXX CUDA)

add_library(cudaKDTree INTERFACE)
target_sources(cudaKDTree INTERFACE
  cukd/common.h
  # iw, sep 22, 2024 - intentionally renamed from cukd/math.h to cukd/cukd-math.h to 
  # avoid name conflicts with system math.h if anybody adds cukd/ to include path
  cukd/cukd-math.h
  cukd/box.h
  cukd/builder.h
  cukd/builder_bitonic.h
  cukd/builder_thrust.h
  cukd/builder_inplace.h
  # SPATIAL k-d tree, with planes at arbitrary locations 
  cukd/spatial-kdtree.h
  cukd/fcp.h
  cukd/knn.h
  )
target_include_directories(cudaKDTree INTERFACE ${PROJECT_SOURCE_DIR}/)

include_directories(/usr/include/eigen3)
aux_source_directory(.  DIR_LIB)
add_executable(icp main.cpp ${DIR_LIB})
target_link_libraries(icp cudaKDTree)
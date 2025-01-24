#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(BATCH_RASTERIZER OFF)

if (BATCH_RASTERIZER)
	add_library(CudaBatchRasterizer
		include/auxiliary.h
		include/forward.h
		include/rasterizer_impl.h
		include/rasterizer.h
		src/forward.cu
		src/rasterizer_impl.cu
	)
	set_target_properties(CudaBatchRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86;89")
	target_include_directories(CudaBatchRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
	target_include_directories(CudaBatchRasterizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
	add_library(CudaRasterizer
		include/auxiliary.h
		include/forward.h
		include/rasterizer_impl.h
		include/rasterizer.h
		src/forward.cu
		src/rasterizer_impl.cu
	)
	set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86;89")
	target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
	target_include_directories(CudaRasterizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()


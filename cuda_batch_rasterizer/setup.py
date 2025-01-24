import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)


def _is_cuda() -> bool:
    return torch.version.cuda is not None


ext_modules = []

# Compiler flags.
# Accelerate compilation speed and prevent compiler from performing negative optimizations!
CXX_FLAGS = ["-g", "-O1", "-std=c++17"]
NVCC_FLAGS = ["-O1", '-Xptxas="-O1"', "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "../third_party/glm/")]

# Include directory
include_dir = os.path.join(ROOT_DIR, 'include')
CXX_FLAGS.append(f'-I{include_dir}')
NVCC_FLAGS.append(f'-I{include_dir}')

def glob(pattern: str):
    root = Path(__name__).parent
    return [str(p) for p in root.glob(pattern)]


if _is_cuda() and CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

if _is_cuda():
    ext_modules.append(
        CUDAExtension(
            name="vanilla_batch_gaussian_splatting",
            sources=["src/rasterizer_impl.cu",
                    "src/forward.cu",
                    "batch_rasterize_points.cu",
                    "ext.cpp"],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
        )
    )

setup(name="vanilla_batch_gaussian_splatting",
      version="0.1.0",
      ext_modules=ext_modules,
      cmdclass={"build_ext": BuildExtension}
      )
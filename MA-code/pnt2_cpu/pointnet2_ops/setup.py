# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ext_src_root = os.path.join(BASE_DIR, "_ext_src")
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

setup(
    name='pointnet2_cpp',
    ext_modules=[
        CUDAExtension(
            name='pointnet2_cpp._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3","-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O3","-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

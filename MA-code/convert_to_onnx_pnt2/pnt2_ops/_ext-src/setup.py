# This file is modified from the original project.
# It will install the custom operators and generate a shared library to register these operators with TorchScript

from setuptools import setup,Extension,find_packages
# from torch.utils import cpp_extension
# setup(
#     name="pnt2_ops",
#     ext_modules=[cpp_extension.CppExtension('pnt2_ops',['try_custom_fps.cpp'],
#                 include_dirs=[])],
#     cmdclass={'build_ext':cpp_extension.BuildExtension}
# )


import glob
import os
import os.path as osp

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = this_dir
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

#exec(open(osp.join("pointnet2_ops", "_version.py")).read())

os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
setup(
    name="pnt2_ops",
    #version=__version__,
    #author="Erik Wijmans",
    #packages=find_packages(),
    #install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pnt2_cuda_ops",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    #include_package_data=True,
)
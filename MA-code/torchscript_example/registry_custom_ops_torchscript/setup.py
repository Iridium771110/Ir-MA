from setuptools import setup,Extension
from torch.utils import cpp_extension
setup(
    name="test_fps_1",
    ext_modules=[cpp_extension.CppExtension('fps_test',['try_custom_fps.cpp'],
                include_dirs=[])],
    cmdclass={'build_ext':cpp_extension.BuildExtension}
)
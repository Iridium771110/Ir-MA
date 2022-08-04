This category shows the example to register a custom operator with TorchScript

To install the custom operator in try_custom_fps.cpp file with TorchScript run the command:

python setup.py install

Then the shared library is in 
build/lib.linux-x86_64-3.7/fps_test.cpython-37m-x86_64-linux-gnu.so

Using torch.ops.load_library(path_to_lib) to load the shared library and use in python file

run the command:

python test.py

to generate the onnx model and find it in this category
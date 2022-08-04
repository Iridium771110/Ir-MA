This category includes the source files for the shared library
This is one of feasible ways to register the custom operators with ONNX Runtime
Through loading the shared library, the execution in python is supported

run command:

mkdir build
cd build
cmake ..
make

to generate the shared library
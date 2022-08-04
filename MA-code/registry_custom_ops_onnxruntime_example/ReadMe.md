This category tests the example onnx model with fps custom operator, this model has only the fps operator in forward function
The test runs under ONNX Runtime
testmodel.onnx is the example onnx model generated from torchscript_example/registry_custom_ops_torchscript

run commands:

mkdir build
cd build
cmake ..
make
./test_fps_onnx ../testmodel.onnx

to see the behavior of the example onnx model
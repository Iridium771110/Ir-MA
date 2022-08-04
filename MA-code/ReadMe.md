This is the repository for the master thesis "Inference Acceleration of point-based neural networks"
This repository includes 6 categories

Category "torchscript_example" shows an example for the converting and test of PointNet model with PyTorch and ONNX, and an example for registration a custom operator with TorchScript.
| Folder "pointnet_model" includes converting and test of PointNet, the original project is cloned from https://github.com/fxia22/pointnet.pytorch 
|| Clone this project, train the PyTorch model and save the trained model. This model will be used in the example for converting
| Folder "registry_custom_ops_torchscript" shows a way to generate an ONNX model with a custom operator
|| In fact, the specific implementation of custom operators with TorchScript during the converting is not necessary, but the information of input and output must be ensured
|| A feasible way to register the custom operator with TorchScript is shown in this folder

Category "registry_custom_ops_onnxruntime_example" shows an example for inference test of ONNX model with custom operator under ONNX Runtime. The ONNX model is generated from the registration example in category "torchscript_example". 

Category "pnt2_cpu" shows the benchmark test. It test the inference cost of PointNet++ with PyTorch.
The original PointNet++ project is cloned from https://github.com/erikwijmans/Pointnet2_PyTorch
Clone this project, train the PyTorch model and save the trained model. And make the benchmark test using the trained model
| Folder "pointnet2_ops" includes the modified custom operators for the test with PyTorch

Category "convert_to_onnx_pnt2" shows the way to convert the PyTorch model to ONNX model
The PointNet++ model is rewriten here and some codes are modified to make the convertor happy
| Folder "pnt2_ops" includes the custom operators from the original project
|| A feasible way to register the custom operator with TorchScript is shown in this folder

Category "quantization" shows the quantization investigation using the methods provided by ONNX Runtime
| Folder "for_py_ops" includes the modified custom operators for execution ONNX Runtime in python
|| A feasible way to register the custom operators with ONNX Runtime is shown in this folder

Category "onnxruntime_inference" shows the inference experiment and the promotion of custom operators
The custom operators are implemented with ONNX Runtime in C++
A feasible way to register the custom operators with ONNX Runtime is shown in this category
| Folder "data" includes the ONNX model and data for inference test
| Folder "kd_tree" includes the fundemental files for k-d tree-based algorithm
| Folder "octree" includes the fundemental files for octree-based algorithm

The test enviroment:

GCC/G++         9.4.0
CUDA            11.1
cuDNN           8.2.1
Python          3.7.11
PyTorch         1.9.0
ONNX            1.10.2
ONNX Runtime    1.10.0
OpenMP          4.5

PyTorch, Python, ONNX, and ONNX Runtime can be installed through anaconda using pip install
OpenMP is usually fixed with the compiler
But the ONNX Runtime library for cpp should complied from the source project or download the complied lib with the corresponding version
Here the used library is "onnxruntime-linux-x64-1.10.0", which can be found on https://github.com/microsoft/onnxruntime/releases
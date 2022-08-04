This category quantizes the ONNX model through dynamic and static quantization method under onnxruntime
In onnxruntime lib, file quantize.py, and in file onnxruntime_inference_collection.py, registration information should be added manually through load the shared library, which is generated for the execution onnxruntime session in python:

sess_option.register_custom_ops_library(path_to_lib)

Once the shared library is generated, run the command:

python quantize.py

to quantize the ONNX model through dynamic and static method
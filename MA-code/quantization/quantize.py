# This file quantize the ONNX model through dynamic and static methods provided by ONNX Runtime

import onnx
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader

# load the data and label
data_p=np.fromfile("test_data.bin",dtype=np.float32) # please modify the path to data
data_p=data_p.reshape(2468,4096,-1)
print(data_p.shape)
labels=np.fromfile("test_label.bin",dtype=np.int64) # please modify the path to target label
labels=labels.reshape(2468,1)
print(labels.shape)

# define a calibration data set for static method
class DataReader(CalibrationDataReader):
    def __init__(self,all_data):
        self.data=all_data
        self.input_name="points"
        self.enum_data_dicts=[]
        self.data_list=[]
        for i in range(0,10):
            frame_data=self.data[i,:,:].copy()
            frame_data=frame_data.reshape(1,4096,6)
            self.data_list.append(frame_data.copy())
        self.enum_data_dicts=iter([{self.input_name:data} for data in self.data_list])

    def get_next(self):  
        return next(self.enum_data_dicts,None)

model_f="script_model.onnx" # please modify the path to original ONNX model
model_d="quantized_model.onnx" # please modify the path for dynamic quantized model
model_q="quantized_model_s.onnx" # please modify the path for static quantized model

quantize_dynamic(model_f,model_d) # an example to do dynamic quantization with default configuration

dr=DataReader(data_p)
quantize_static(model_f,model_q,optimize_model=False,calibration_data_reader=dr) # an example to do static quantization with default configuration

# check the quantized model and observe the compute graph
onnx_model=onnx.load(model_d)
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('invalid: %s' % e)
else:
    print(onnx.helper.printable_graph(onnx_model.graph))
    print("well")

# test the inference under ONNX Runtime in python
import onnxruntime

session_options= onnxruntime.SessionOptions()
session_options.register_custom_ops_library("/home/dong/WS/test/for_py_ops/build/libcustom_ops.so") # please modify the path to the shared library which registers the custom operators with ONNX Runtime
ort_session=onnxruntime.InferenceSession("script_model.onnx",session_options) # please modify the path to the model for inference
example=data_p[1,:,:]
example=example.reshape(1,4096,6) # generate an example input
print(example.shape)
out=ort_session.run(None,{"points":example}) # test the inference for single example
print(out)
import onnx
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader


# data_p=np.fromfile("test_data.bin",dtype=np.float32)
# data_p=data_p.reshape(2468,4096,-1)
# print(data_p.shape)
# labels=np.fromfile("test_label.bin",dtype=np.int64)
# labels=labels.reshape(2468,1)
# print(labels.shape)

# class DataReader(CalibrationDataReader):
#     def __init__(self,all_data):
#         self.data=all_data
#         self.input_name="points"
#         self.enum_data_dicts=[]
#         self.data_list=[]
#         for i in range(0,10):
#             frame_data=self.data[i,:,:].copy()
#             frame_data=frame_data.reshape(1,4096,6)
#             self.data_list.append(frame_data.copy())
#         self.enum_data_dicts=iter([{self.input_name:data} for data in self.data_list])
#     def get_next(self):
        
#         return next(self.enum_data_dicts,None)

model_f="script_model.onnx"
model_q="quantized_model_s.onnx"


# quantize_dynamic(model_f,model_q)

# dr=DataReader(data_p)

#quantize_static(model_f,model_q,optimize_model=False,calibration_data_reader=dr)

onnx_model=onnx.load(model_f)
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('invalid: %s' % e)
else:
    print(onnx.helper.printable_graph(onnx_model.graph))
    print("well")

# import onnxruntime

# onnx_model=onnx.load(model_f)
# onnx.checker.check_model(onnx_model)

# session_options= onnxruntime.SessionOptions()
# #session_options.register_custom_ops_library("/home/dong/WS/test/for_py_ops/build/libcustom_ops.so")
# print("a")
# ort_session=onnxruntime.InferenceSession("script_model.onnx",session_options)
# print("b")
# example=data_p[1,:,:]
# example=example.reshape(1,4096,6)
# print(example.shape)
# out=ort_session.run(None,{"points":example})
# print(out)
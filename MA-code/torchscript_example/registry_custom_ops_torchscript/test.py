import torch
# This file converts an example model using fps operator from PyTorch model to ONNX model
# This method first converts the model to TorchScript model and then to ONNX model, so the registration of fps operator with TorchScript is necessary

torch.ops.load_library(
    "/home/dong/WS/arrange/torchscript_example/registry_custom_ops_torchscript/build/lib.linux-x86_64-3.7/fps_test.cpython-37m-x86_64-linux-gnu.so") # load the library, please modify the path to lib

example=torch.randn(1,5,3)
print(example)
example=example.transpose(2,1)
output=torch.zeros([1,3,5],dtype=torch.int64) # generate example input for the operator

def test_fps_registry(g,input_data,output_data,a):
    return g.op("custom_ops_node::test_fps_node",input_data, output_data,a,epsilon_i=10) # specify the called onnxruntime operator's name
    # specify inputs and the external parameter "epsilon", "_i" means the data type int
from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('custom_ops::test_fps', test_fps_registry, 10) # symbolic registration for TorchScript operator

print(torch.ops.custom_ops.test_fps(example,output,50)) # test the TorchScript operator (loaded library)

class test_model(torch.nn.Module):
    # define an example model using fps operator in forward function
    def __init__(self):
        super(test_model,self).__init__()
        self.module_1=torch.nn.Conv1d(5,4,1)
        self.module_2=torch.nn.Conv1d(3,6,1)
    
    def forward(self,in1,out1):
        # type: ( torch.Tensor, torch.Tensor) -> torch.Tensor
        # x=in1.transpose(2,1)
        # x=self.module_1(x)
        # x=x.transpose(2,1)
        i=50.0 # to test the scalar transmission
        x=torch.ops.custom_ops.test_fps(in1,out1,i)
        # x=self.module_2(x)
        # x=x.permute(0,2,1)
        return x

model=test_model()
print(model(example,output))

scripted=torch.jit.script(model) # generate torchscript model through script method

example_out=scripted(example,output) # generate the example output by torchscript model

torch.onnx.export(
    scripted,
    (example,output),
    "testmodel.onnx",
    opset_version=10,
    input_names=["original","to_sample"],
    output_names=["sampled"],
    custom_opsets={"custom_ops_node":10},
    example_outputs=example_out
) # export the example onnx model with fps operator
print("well")
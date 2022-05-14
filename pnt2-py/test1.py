from torch.autograd import Function
import torch
# class NM(Function):
#     @torch.jit.script_method
#     @staticmethod
#     def forward(ctx, indata):
#         return indata
    
#     @staticmethod
#     def backward(ctx, graddata):
#         return ()
# NMD=NM.apply
def NMD(indata):
    return indata
class TMD(torch.nn.Module):
    def __init__(self):
        super(TMD,self).__init__()
        self.module=torch.nn.Conv1d(1,1,1)
    
    def forward(self,in1):
        x=self.module(in1)
        x=NMD(x)
        return x

t=torch.randn(1,1,10)
m=TMD()
print(m(t))

t_m=torch.jit.trace(m,t)
t_m.save('tmd.pt')

class Model(torch.nn.Module):
    def forward(self, x):
        xt = x.transpose(1,2)
        return xt

m  = Model().cuda()
i = (torch.randn(1,2,3).cuda(),)

torch.onnx.export(m, i, "model.onnx",
                  input_names=["INPUT_0"],
                  output_names=["OUTPUT_0"],
                  dynamic_axes={"INPUT_0": {0: "batch_size"},
                                "OUTPUT_0": {0: "batch_size"}})
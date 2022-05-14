import torch

torch.ops.load_library(
    "/home/dong/WS/test/pnt2_ops/_ext-src/build/lib.linux-x86_64-3.7/pnt2_cuda_ops.cpython-37m-x86_64-linux-gnu.so")
def sampling_registry(g,points_xyz,sample_num): #对points_xyz采样sample_num个点，返回为对应index
    #(B, N, 3) tensor,int ->(B, s_n) tensor
    return g.op("onnx_pnt2_ops::onnx_sampling",points_xyz, sample_num) #指定运算子调用运算节点
def gather_points_registry(g,points_xyz,gather_index): #对points_xyz取对应gather_index的点，返回为所取点
    #(B, C, N) tensor,(B, s_n) tensor ->(B, C, s_n) tensor
    return g.op("onnx_pnt2_ops::onnx_gather_points",points_xyz, gather_index) #指定运算子调用运算节点
def ball_query_registry(g,centers_xyz,points_xyz,radius,sample_num): #对points_xyz以centers_xyz为中心以radius为半径取最多sample_num个点，返回为对应index
    #(B, c_n, 3) tensor,(B, N, 3) tensor,float,int->(B, c_n, s_n) tensor
    return g.op("onnx_pnt2_ops::onnx_ball_query",  centers_xyz,points_xyz,radius, sample_num) #指定运算子调用运算节点
def grouping_registry(g,full_features_map,grouping_index): #对full_features_map按照grouping_index取对应点，返回为所取点（特征）
    #(B, C, N) tensor,(B, c_n, s_n) tensor ->(B, C, c_n, s_n) tensor
    return g.op("onnx_pnt2_ops::onnx_grouping",full_features_map,grouping_index) #指定运算子调用运算节点

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('pnt2_ops::sampling', sampling_registry, 11) #指定语义同名函数
register_custom_op_symbolic('pnt2_ops::gather_points', gather_points_registry, 11) #指定语义同名函数
register_custom_op_symbolic('pnt2_ops::ball_query', ball_query_registry, 11) #指定语义同名函数
register_custom_op_symbolic('pnt2_ops::grouping', grouping_registry, 11) #指定语义同名函数


class NMD(torch.nn.Module):
    def __init__(self):
        super(NMD,self).__init__()
        self.r=1.0
        self.s=5
        self.i=1
    
    def forward(self,p):
        xyz=p[:,:,0:3].contiguous()
        
        s=torch.ops.pnt2_ops.sampling(xyz,self.s)
        p=p.transpose(1,2).contiguous()
        gp=torch.ops.pnt2_ops.gather_points(p,s)
        c=torch.ops.pnt2_ops.gather_points(xyz,s)
        print(xyz.dtype)
        b=torch.ops.pnt2_ops.ball_query(xyz,xyz,self.r,self.i)
        #g=torch.ops.pnt2_ops.grouping(p,b)
        return b

model=NMD().cuda()
example_input=torch.randn(1,5,5).cuda()
example_input[0,0,0]=1.2987
example_input[0,0,1]=-0.9477
example_input[0,0,2]=0.6843
example_input[0,0,3]=0.3944
example_input[0,0,4]=2.0187
example_input[0,1,0]=0.9125
example_input[0,1,1]=0.9375
example_input[0,1,2]=0.2431
example_input[0,1,3]=-0.3821
example_input[0,1,4]=0.0988
example_input[0,2,0]=2.4182
example_input[0,2,1]=1.7426
example_input[0,2,2]=-0.9073
example_input[0,2,3]=1.1238
example_input[0,2,4]=-1.2532
example_input[0,3,0]=1.6646
example_input[0,3,1]=-0.2011
example_input[0,3,2]=-1.0471
example_input[0,3,3]=0.9630
example_input[0,3,4]=0.4516
example_input[0,4,0]=0.8094
example_input[0,4,1]=0.7476
example_input[0,4,2]=-0.4819
example_input[0,4,3]=1.0698
example_input[0,4,4]=-2.2114

print(example_input)
example_output=model(example_input)
print(example_output)

scripted_model=torch.jit.script(model)
scripted_model.save('scripted_pnt2_cls_ssg.pt')
reload_model=torch.jit.load('scripted_pnt2_cls_ssg.pt')

print(reload_model(example_input))

torch.onnx.export(
    reload_model,
    example_input,
    "script_model.onnx",
    opset_version=11,
    input_names=["points"],
    output_names=["results"],
    custom_opsets={"onnx_pnt2_ops":11},

    example_outputs=example_output,
    dynamic_axes={"points":{0:"batch_size",1:"num_points"},"results":{0:"batch_size"}}
    )

import onnx
onnx_model=onnx.load("script_model.onnx")
onnx.checker.check_model(onnx_model)    
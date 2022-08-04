# This file converts the trained model first from PyTorch to TorchScript and then to ONNX

import pnt2_tr # validation function is used to check if the process runs normally
import argparse
import torch
from pnt2_data import ModelNet40Cls
from pnt2_cls_ssg import pnt2_cls_ssg

def write2bin(points,target,op):
    # This function writes the data into a bin-file, which will be used as the inference data under ONNX Runtime

    import numpy as np
    pp=points.cpu()
    pp=pp.numpy()
    bin_writer=open("test_data.bin",op)
    pp=np.ascontiguousarray(pp)
    bin_writer.write(pp)
    bin_writer.close()
    bin_writer=open("test_label.bin",op)
    pp=target.cpu()
    pp=pp.numpy()
    pp=np.ascontiguousarray(pp)
    bin_writer.write(pp)
    bin_writer.close()

if  __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/home/dong/WS/Pointnet2_PyTorch/pointnet2/data", help='Root to the dataset') # please modify the path to data
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_points', type=int, default=4096, help='Number of the training points')
    parser.add_argument('--model_use_xyz', type=bool, default=True, help='unknown usage')
    parser.add_argument('--gpus', type=str, default='0', help='Cuda ids')
    parser.add_argument('--optimizer_lr', type=float, default=1e-3, help='Initial learing rate')
    parser.add_argument('--optimizer_lr_decay', type=float, default=0.7, help='Initial learing rate decay')
    parser.add_argument('--optimizer_decay_step', type=float, default=50.0, help='Initial learing rate decay step')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.0, help='Initial weight decay')
    parser.add_argument('--total_epoch', type=int, default=50, help='Number of traing epoches')
    args = parser.parse_args()
    
    device_ids = list(map(int, args.gpus.strip().split(','))) if ',' in args.gpus else [int(args.gpus)]
    print(args.num_points) # check the configuration

#   dataloader
#   copied from pnt2_data.py, reserve the necessary parts
    train_transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudScale(),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudRotatePerturbation(),
                d_utils.PointcloudTranslate(),
                d_utils.PointcloudJitter(),
                d_utils.PointcloudRandomInputDropout(),
            ]
        )
    va_dataset = ModelNet40Cls(
        args.num_points, 
        transforms=None, 
        train=False
    )
    va_dataloader = DataLoader(
        va_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    print('Test set: {}'.format(len(va_dataset)))

#   load the PyTorch model and do evaluation once to check if the model loaded correctly
    tr_model=pnt2_cls_ssg(args)
    tr_model.load_state_dict(torch.load('pnt2_cls_ssg.pth'))

    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    print("Using device: ",device )
    tr_model = tr_model.to(device)
    tr_model.eval()

    loss,num_correct,num_total,acc=pnt2_tr.epoch_va(0, va_dataloader, tr_model, device)
    print("pseudo test, accurancy: ",acc)

#   write the data into a bin-file
    # op="wb"
    # num=0
    # for data,labels in va_dataloader:
    #     points=data.cpu()
    #     target=labels.cpu()
    #     write2bin(points, target, op)
    #     num+=1
    #     op="ab"
    # print(data.size(),' ',labels.size(),' ',num)
    
#   convert the model to TorchScript model
    example_input=torch.randn(args.batch_size,args.num_points,3+3).cuda() # generate an example input
    # traced_model=torch.jit.trace(tr_model,example_input)
    # traced_model.save('traced_pnt2_cls_ssg.pt') # convert the model using tracing
    scripted_model=torch.jit.script(tr_model)
    scripted_model.save('scripted_pnt2_cls_ssg.pt') #convert the model using script. The converted model from this method performs usually better than tracing. It is recommended.

#   check if the converted TorchScript model performs as well as the PyTorch model. 
#   A similar precision means a successful converting
    # reload_model=torch.jit.load('traced_pnt2_cls_ssg.pt')
    # loss,num_correct,num_total,acc=pnt2_tr.epoch_va(0, va_dataloader, reload_model, device)
    # for data,labels in va_dataloader:
    #     print(data.size(),' ',labels.size())
    #     break
    # print("pseudo test, accurancy: ",acc) # precision checking for traced model

    reload_model=torch.jit.load('scripted_pnt2_cls_ssg.pt')
    loss,num_correct,num_total,acc=pnt2_tr.epoch_va(0, va_dataloader, reload_model, device)
    for data,labels in va_dataloader:
        print(data.size(),' ',labels.size())
        break
    print("pseudo test, accurancy: ",acc) # precision checking for scripted model
    
#   convert the TorchScript model to ONNX model
    example_output=reload_model(example_input) # generate an example output, which is required in ONNX converting

    def sampling_registry(g,points_xyz,sample_num):
        #(B, N, 3) tensor,int ->(B, s_n) tensor
        return g.op("onnx_pnt2_ops::onnx_sampling",points_xyz, sample_num) # specify the called onnxruntime operator's name
    def gather_points_registry(g,points_xyz,gather_index):
        #(B, C, N) tensor,(B, s_n) tensor ->(B, C, s_n) tensor
        return g.op("onnx_pnt2_ops::onnx_gather_points",points_xyz, gather_index) # specify the called onnxruntime operator's name
    def ball_query_registry(g,centers_xyz,points_xyz,radius,sample_num): 
        #(B, c_n, 3) tensor,(B, N, 3) tensor,float,int->(B, c_n, s_n) tensor
        return g.op("onnx_pnt2_ops::onnx_ball_query", centers_xyz,points_xyz, radius, sample_num) # specify the called onnxruntime operator's name
    def grouping_registry(g,full_features_map,grouping_index): 
        #(B, C, N) tensor,(B, c_n, s_n) tensor ->(B, C, c_n, s_n) tensor
        return g.op("onnx_pnt2_ops::onnx_grouping",full_features_map,grouping_index) # specify the called onnxruntime operator's name

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('pnt2_ops::sampling', sampling_registry, 11) # symbolic registration for TorchScript operator
    register_custom_op_symbolic('pnt2_ops::gather_points', gather_points_registry, 11) # symbolic registration for TorchScript operator
    register_custom_op_symbolic('pnt2_ops::ball_query', ball_query_registry, 11) # symbolic registration for TorchScript operator
    register_custom_op_symbolic('pnt2_ops::grouping', grouping_registry, 11) # symbolic registration for TorchScript operator

    torch.onnx.export(
    reload_model,
    example_input,
    "script_model.onnx",
    opset_version=13,
    input_names=["points"],
    output_names=["results"],
    custom_opsets={"onnx_pnt2_ops":11},

    example_outputs=example_output,
    dynamic_axes={"points":{0:"batch_size",1:"num_points"},"results":{0:"batch_size"}}
    ) # export the onnx model

#   check the converted onnx model
    import onnx
    onnx_model=onnx.load("script_model.onnx")
    import numpy as np
    a=np.fromfile("test_label.bin",dtype=np.int32)
    print(a.shape)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('invalid: %s' % e)
    else:
        print(onnx.helper.printable_graph(onnx_model.graph)) # check the compute graph
        print("well")
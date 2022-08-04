# This file is an example to convert the TorchScript model into the ONNX model without custom operators
# The Dataloader relies on the original PointNet project
# The test combines the PointNet model and the custom operator realizing furthest point sampling

import onnx
import onnxruntime
import torch
import numpy

# load the TorchScript model and convert it to an ONNX model
model_traced_pnt=torch.jit.load('cls_script_model.pt') # please modify the path to model
example=torch.randn(3,3,2500)
example_np=example.numpy()
example=example.cuda()
example_out=model_traced_pnt(example)
onnx_path="cls_onnx.onnx"
torch.onnx.export(model_traced_pnt,
                  example,onnx_path,
                  opset_version=10,
                  input_names=["in"],
                  output_names=["out"],
                  example_outputs=example_out,
                  dynamic_axes={"in":[0,2],"out":[0]}
                )

# load the ONNX model and set the onnxruntime session
onnx_model=onnx.load("cls_onnx.onnx")
onnx.checker.check_model(onnx_model)
ort_session=onnxruntime.InferenceSession("cls_onnx.onnx")


import argparse
import random
import torch
import torch.nn.parallel
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from tqdm import tqdm

# initial configuration
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=16, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--dataset', type=str, default='../../data', help="dataset path") # please modify the path to data
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

# set the data loader
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')

# test the data loader
testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(test_dataset))
num_classes = len(test_dataset.classes)
num_classes=16
print('classes', num_classes)

# comaprison test for ONNX model and TorchScript model
classifier = model_traced_pnt
total_correct = 0
total_testset = 0
total_correct_onnx = 0
torch.ops.load_library(
    "/home/dong/WS/deploy_pointnet/try_script/build/lib.linux-x86_64-3.7/fps_test.cpython-37m-x86_64-linux-gnu.so") # please modify the path to the shared library
# This library is for the custom operator furthest point sampling, which is the example in the category registry_custom_ops_torchscript

for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    b_size=torch.tensor([points.size()[0]],dtype=torch.int32)
    out=torch.randn(b_size,3,500) # set the sample number=500 for FPS

    points, target = points.cuda(), target.cuda()
#   test the inference with TorchScript
    pred = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]
#   test the inference with ONNX and ONNX Runtime, combined with FPS operator
    points=points.cpu()
    points=torch.ops.custom_ops.test_fps(points,out) # using FPS to downsample points
    data_onnx=points.numpy()
    pred_onnx=ort_session.run(None,{"in":data_onnx})
    pred_choice_onnx=pred_onnx[0].argmax(1)
    target=target.cpu()
    correct_onnx=numpy.sum(pred_choice_onnx==target.data.numpy())
    total_correct_onnx += correct_onnx.item()
    
# print the results
print("final accuracy {}".format(total_correct / float(total_testset)))
print("onnx final accuracy {}".format(total_correct_onnx / float(total_testset)))





# This file is for generation of a TorchScript model from a PyTorch model without custom operators
# relies on the original PointNet project
import argparse
import random
import torch
import torch.nn.parallel
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls
from tqdm import tqdm

# def write2bin(points,target,op):
#     # write the data into a bin file for test
#     import numpy as np
#     pp=points.cpu()
#     pp=pp.numpy()
#     bin_writer=open("test_data.bin",op)
#     pp=np.ascontiguousarray(pp)
#     bin_writer.write(pp)
#     bin_writer.close()
#     bin_writer=open("test_target.bin",op)
#     pp=target.cpu()
#     pp=pp.numpy()
#     pp=np.ascontiguousarray(pp)
#     bin_writer.write(pp)
#     bin_writer.close()

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
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='../../data1', help="dataset path") # please modify the path to data
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

# set the test data loader
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

# load the saved PyTorch model
classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
classifier.load_state_dict(torch.load('cls_model_200.pth')) # please modify the path to model
classifier.cuda()

# verify the loaded PyTorch model
total_correct = 0
total_testset = 0

#op="wb"
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]
    #write2bin(points,target,op)
    #op="ab"

print("final accuracy {}".format(total_correct / float(total_testset)))

# generate an example input and convert the model into TorchScript model
batch_size=16
k=3
example=torch.rand(batch_size,k,opt.num_points)
example=example.cuda()
traced_model=torch.jit.trace(classifier,example) # the other way is torch.jit.script()
traced_model.save('cls_script_model_2.pt')
reload_model=torch.jit.load('cls_script_model_2.pt')
print(reload_model)

print(reload_model(points))
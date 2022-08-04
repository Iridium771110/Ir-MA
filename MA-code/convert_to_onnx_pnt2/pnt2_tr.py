# This file is a rewriten training file for SSG model
# It is more simple to read than the original project

from pnt2_cls_ssg import pnt2_cls_ssg
from pnt2_data import ModelNet40Cls
import torch.nn.functional as F

import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched

def epoch_tr(index_epoch,tr_dataloader,tr_model,optimizer,device):
    # This function trains the model for each epoch

    #print("training epoch : ", index_epoch)
    tr_model.train()
    num_correct,num_total=0,0
    losses=[]
    for data, labels in tr_dataloader:
        data=data.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        result=tr_model(data)
        loss=F.cross_entropy(result, labels)
        loss.backward()
        optimizer.step()

        re_labels=torch.max(result,dim=-1)[1]
        num_correct+=torch.sum(re_labels==labels)
        num_total+=data.shape[0]
        losses.append(loss.item())
        acc=float(num_correct)/float(num_total)
    return np.mean(losses), num_correct, num_total, acc

def epoch_va(index_epoch,va_dataloader,tr_model,device):
    # This function validates the model for each epoch

    #print("validation epoch : ", index_epoch)
    tr_model.eval()
    num_total, num_correct=0,0
    losses=[]
    for data,labels in va_dataloader:
        with torch.no_grad():
            data=data.to(device)
            labels=labels.to(device)
            result=tr_model(data)
            loss=F.cross_entropy(result,labels)

            re_labels=torch.max(result,dim=-1)[1]
            num_correct+=torch.sum(re_labels==labels)
            num_total+=data.shape[0]
            losses.append(loss.item())
            acc=float(num_correct)/float(num_total)
    return np.mean(losses), num_correct, num_total, acc

def train(total_epoch,log_path,tr_model,lr_scheduler,tr_dataloader,va_dataloader,optimizer,device):
    # This function trains the model and finally save the trained model

    for index_epoch in range(total_epoch):
        loss,num_correct,num_total,acc=epoch_tr(index_epoch,tr_dataloader,tr_model,optimizer,device)
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        print('Train Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'
            .format(index_epoch, total_epoch, lr, loss, num_correct, num_total, acc))

        loss,num_correct,num_total,acc=epoch_va(index_epoch,va_dataloader,tr_model,device)
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        print('Validation Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'
            .format(index_epoch, total_epoch, lr, loss, num_correct, num_total, acc))

        lr_scheduler.step()

    torch.save(tr_model.state_dict(), '/home/dong/WS/test/cls_model_%d.pth' % (total_epoch-1)) # please modify the path
    
if  __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/home/dong/WS/Pointnet2_PyTorch/pointnet2/data", help='Root to the dataset')
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
    print(args.num_points)

#   dataloader
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
    tr_dataset = ModelNet40Cls(
        args.num_points, 
        transforms=train_transforms, 
        train=True
    )
    va_dataset = ModelNet40Cls(
        args.num_points, 
        transforms=None, 
        train=False
    )
    tr_dataloader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    va_dataloader = DataLoader(
        va_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    print('Train set: {}'.format(len(tr_dataset)))
    print('Test set: {}'.format(len(va_dataset)))

#   set configuration for training
    tr_model=pnt2_cls_ssg(args)
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    print("Using device: ",device )
    tr_model = tr_model.to(device)

    lr_clip = 1e-5
    optimizer = torch.optim.Adam(
                tr_model.parameters(),
                lr=args.optimizer_lr,
                weight_decay=args.optimizer_weight_decay,
            )

    lr_lbmd = lambda index_epoch: max(
                args.optimizer_lr_decay
                ** (
                    int(
                        #tr_model.global_step
                        index_epoch
                        * args.batch_size
                        / args.optimizer_decay_step
                    )
                ),
                lr_clip / args.optimizer_lr,
            )
    lr_scheduler=lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
    print(args.optimizer_decay_step)
    log_path="/home/dong/WS/test/"

    train(args.total_epoch,log_path,tr_model,lr_scheduler,tr_dataloader,va_dataloader,optimizer,device)
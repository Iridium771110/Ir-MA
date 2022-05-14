import torch
import torch.nn as nn
#import torch.nn.functional as F
from pointnet2_modules import PointnetFPModule, PointnetSAModule

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class pnt2_cls_ssg(nn.Module):
    def __init__(self,hparams):
        super(pnt2_cls_ssg,self).__init__()
        self.hparams = hparams
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=self.hparams.model_use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams.model_use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=self.hparams.model_use_xyz
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        #features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        features=pc[..., 3:].transpose(1, 2).contiguous()
        # if pc.size(-1)>3 :
        #     features=pc[..., 3:].transpose(1, 2).contiguous()
        #     #a=pc[..., 3:].transpose(1, 2).contiguous()
        # else:
        #     features=None
        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        # for module in self.SA_modules:
        #     xyz, features = module(xyz, features)
        xyz,features=self.SA_modules[0](xyz,features)
        xyz,features=self.SA_modules[1](xyz,features)
        aa,features=self.SA_modules[2](xyz,features)

        return self.fc_layer(features.squeeze(-1))


if  __name__ == "__main__":
    cfg={'model.use_xyz':True}
    model = pnt2_cls_ssg(cfg).to(device)
    print(model)

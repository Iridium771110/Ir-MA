# This file defines the modules used in PointNet++
# This file is copied from the original project and modified for converting
# For converting, the data type interpretation will be concerned even it is commented out

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2_utils

torch.ops.load_library(
    "/home/dong/WS/test/pnt2_ops/_ext-src/build/lib.linux-x86_64-3.7/pnt2_cuda_ops.cpython-37m-x86_64-linux-gnu.so") # please modify the path to library

def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    # This module is modified for converting

    def __init__(self,n_s):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.n_s=n_s

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        a=torch.zeros(1,1,1) # used as a placeholder, to ensure the data type consistency
        new_xyz = (
            # pointnet2_utils.gather_operation(
            #     xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            # )
            # .transpose(1, 2)
            # .contiguous()

            # torch.ops.pnt2_ops.gather_points(
            #     xyz_flipped, torch.ops.pnt2_ops.sampling(xyz, self.npoint)
            # )
            # .transpose(1, 2)
            # .contiguous()
             torch.ops.pnt2_ops.gather_points(
                xyz_flipped, torch.ops.pnt2_ops.sampling(xyz, self.npoint) 
                # use the registered operators in the shared library rather than the operators in pointnet2_utils.py, which causes unrecognition problem in TorchScript
            )
            .permute(0, 2, 1) # use permute function to replace transpose function. It provides more axis information and is clear for convertor
            .contiguous() 
            if self.npoint is not None
            else a # to replace None for the data type consistency
        )

        for grouper,mlp in zip(self.groupers,self.mlps):
            new_features = grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = mlp(new_features)  # (B, mlp[-1], npoint, nsample)
            
            # kernel_size_1=int(torch.tensor(new_features.size(3)).item())
            # kernel_size_1=new_features.size(3)
            # new_features = F.max_pool2d(
            #     new_features, kernel_size=[1, kernel_size_1]
            # )  # (B, mlp[-1], npoint, 1)
            # new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features=torch.max(new_features,dim=3)[0] # to replace the fuctions above. F.max_pool2d causes trouble in converting
            
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__(nsamples)

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)

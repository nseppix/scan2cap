import torch.nn as nn
import torch

from lib.pointnet2.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG


class PointNetExtractor(nn.Module):
    def __init__(self, pretrain_mode=False, feature_channels=3, use_xyz=True):
        super().__init__()
        self.pretrain_mode = pretrain_mode
        self.feature_channels = feature_channels + 1
        self.use_xyz = use_xyz

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[self.feature_channels, 32, 32, 64], [self.feature_channels, 64, 64, 128], [self.feature_channels, 64, 96, 128]],
                use_xyz=self.use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.use_xyz,
            )
        )

        self.fc_layer_1 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
        )

        self.fc_layer_2 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

    def forward(self, data_dict):
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
        pointcloud = data_dict["point_clouds"]
        xyz, features = self._break_up_pc(pointcloud)

        box_center = data_dict["ref_center_label"]
        box_size = data_dict["ref_size_residual_label"]

        mask = (box_center.unsqueeze(1) - box_size.unsqueeze(1) / 2 <= xyz) & (xyz <= box_center.unsqueeze(1) + box_size.unsqueeze(1) / 2)
        mask = torch.all(mask, dim=2, keepdim=True).permute( 0, 2, 1).to(dtype=torch.float32)
        features = torch.cat([features, mask], dim=1)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        pc_features = self.fc_layer_1(features.squeeze(-1))
        data_dict["ref_obj_features"] = pc_features
        if self.pretrain_mode:
            data_dict["ref_obj_cls_scores"] = self.fc_layer_2(pc_features)
        return data_dict

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features
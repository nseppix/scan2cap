import torch
from torch import nn
import torch.nn.functional as F

import sys, os

import pointnet2_utils
from data.scannet.model_util_scannet import ScannetDatasetConfig

from models.votenet import VoteNet
from models.proposal_module import decode_scores

DC = ScannetDatasetConfig()


class VoteNetWrapperModule(VoteNet):

    def __init__(self, input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps"):
        super().__init__(num_class=DC.num_class, num_heading_bin=DC.num_heading_bin, num_size_cluster=DC.num_size_cluster, mean_size_arr=DC.mean_size_arr, input_feature_dim=input_feature_dim, num_proposal=num_proposal,
                         vote_factor=vote_factor, sampling=sampling)

    def forward(self, data_dict):
        data_dict = self.backbone_net(data_dict)

        # --------- HOUGH VOTING ---------
        xyz = data_dict['fp2_xyz']
        features = data_dict['fp2_features']
        data_dict['seed_inds'] = data_dict['fp2_inds']
        data_dict['seed_xyz'] = xyz
        data_dict['seed_features'] = features

        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict['vote_xyz'] = xyz
        data_dict['vote_features'] = features

        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.pnet.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps':
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(data_dict['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.pnet.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = data_dict['seed_xyz'].shape[1]
            batch_size = data_dict['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.pnet.vote_aggregation(xyz, features, sample_inds)
        else:
            print('Unknown sampling strategy: %s. Exiting!' % (self.sampling))
            exit()
        data_dict['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
        data_dict['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, feature_dim)
        data_dict['aggregated_vote_inds'] = sample_inds  # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal
        data_dict['aggregated_vote_features_mean'] = torch.mean(features, dim=2)

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.pnet.bn1(self.pnet.conv1(features)))
        net = F.relu(self.pnet.bn2(self.pnet.conv2(net)))
        net = self.pnet.conv3(net)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        data_dict = decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        return data_dict
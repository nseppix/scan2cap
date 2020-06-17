from torch import nn

from models.baseline_captioning_module import Decoder
from models.pointnet_extractor_module import PointNetExtractor
from models.votenet_wrapper_module import VoteNetWrapperModule


class Scan2CapModel(nn.Module):
    def __init__(self, vocab_list, embedding_dict, feature_channels=0, use_votenet=False):
        super().__init__()
        self.feature_channels = feature_channels
        self.use_votenet = use_votenet

        self.pn_extractor = PointNetExtractor(feature_channels=feature_channels, pretrain_mode=True)
        if self.use_votenet:
            self.votenet_extractor = VoteNetWrapperModule(feature_channels)
        self.decoder = Decoder(vocab_list=vocab_list, embedding_dict=embedding_dict)

    def forward(self, data_dict):
        data_dict = self.pn_extractor(data_dict)
        if self.use_votenet:
            data_dict = self.votenet_extractor(data_dict)
        data_dict = self.decoder(data_dict)
        return data_dict

    def load_pn_extractor(self, state_dict):
        self.pn_extractor.load_state_dict(state_dict)

    def load_votenet(self, state_dict):
        self.votenet_extractor.load_state_dict(state_dict, strict=False)

    def load_decoder(self, state_dict):
        self.decoder.load_state_dict(state_dict)
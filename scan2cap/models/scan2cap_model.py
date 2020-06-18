from torch import nn

from models.baseline_captioning_module import Decoder
from models.pointnet_extractor_module import PointNetExtractor
from models.votenet_wrapper_module import VoteNetWrapperModule
from models.attention_captioning import Attentive_Decoder


class Scan2CapModel(nn.Module):
    def __init__(self, vocab_list, embedding_dict, feature_channels=0, use_votenet=False, use_attention=False):
        super().__init__()
        self.feature_channels = feature_channels
        self.use_votenet = use_votenet
        self.use_attention = use_attention

        self.pn_extractor = PointNetExtractor(feature_channels=feature_channels, pretrain_mode=False)
        
        if self.use_votenet:
            # Only use xyz + height for now, because pretrained model does not use color or normal info
            self.votenet_extractor = VoteNetWrapperModule(input_feature_dim=1)
            if self.use_attention:
                self.decoder = Attentive_Decoder(vocab_list=vocab_list, embedding_dict=embedding_dict)
        if not self.use_attention:
            self.decoder = Decoder(vocab_list=vocab_list, embedding_dict=embedding_dict, use_votenet=self.use_votenet)
        if self.use_attention and not self.use_votenet:
            raise Exception("Attention can't be used without votenet") 
        
    def forward(self, data_dict):
        data_dict = self.pn_extractor(data_dict)
        if self.use_votenet:
            data_dict = self.votenet_extractor(data_dict) 
        data_dict = self.decoder(data_dict)
        return data_dict

    def load_pn_extractor(self, state_dict):
        self.pn_extractor.load_state_dict(state_dict)

    def load_votenet(self, state_dict):
        self.votenet_extractor.load_state_dict(state_dict)

    def load_decoder(self, state_dict):
        self.decoder.load_state_dict(state_dict)
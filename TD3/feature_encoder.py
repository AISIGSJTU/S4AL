import torch
import torch.nn as nn

from predictor.mobilenetv2 import MobileNetV2


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MobileNetV2()
        self.state_feat_dim = self.encoder.last_channel

    def forward(self, state: torch.Tensor, ) -> torch.Tensor:
        return self.encoder(state)

    def encoder_load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict)

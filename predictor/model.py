import torch
import torch.nn as nn


from predictor.mobilenetv2 import MobileNetV2


class Predictor(nn.Module):
    def __init__(self, num_classes, ckp_path=None):
        super().__init__()
        self.encoder = MobileNetV2()
        if ckp_path:
            self.encoder.load_state_dict(torch.load(ckp_path))
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.last_channel, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

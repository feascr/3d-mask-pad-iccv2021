import torch
from torch import nn


class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, label_smoothing_value, device):
        assert 0.0 < label_smoothing_value < 0.5, 'Label Smoothing value shoud be in (0, 0.5)'
        super(SmoothedBCEWithLogitsLoss, self).__init__()
        self.label_smoothing_value = label_smoothing_value
        self.bce_loss = nn.BCEWithLogitsLoss().to(device)

    def forward(self, output, target):
        smoothed_target = target * (1 - self.label_smoothing_value) + (1 - target) * self.label_smoothing_value
        return self.bce_loss(output, smoothed_target)
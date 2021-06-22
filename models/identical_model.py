import torch
import torch.nn as nn
from .utils import get_backbone
from . import register_model


@register_model('identical')
class Identical(nn.Module):
    def __init__(self, cfg):
        super(Identical, self).__init__()
        self.backbone = get_backbone(cfg, 1, exportable=False)

    def forward(self, x):
        x = self.backbone(x)
        return {'logits': x}
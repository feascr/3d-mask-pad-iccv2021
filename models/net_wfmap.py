import torch
import torch.nn as nn
from .utils import get_backbone
from . import register_model


@register_model('net_wfmap')
class NetWFMap(nn.Module):
    def __init__(self, cfg):
        super(NetWFMap, self).__init__()

        self.backbone = get_backbone(cfg, 1, exportable=False)
        def get_fmap(self):
            def hook(model, input, output):
                self._fmap = output
            return hook

        current_module = self.backbone
        for atr in cfg.backbone.fmap_block_seq:
            try:
                current_module = current_module.__getattr__(atr)
            except:
                current_module = current_module.__getitem__(atr)
        current_module.register_forward_hook(get_fmap(self))
        
        if cfg.use_HSV: # HSV
            inf_channels = 6
        else:
            inf_channels = 3
        
        with torch.no_grad():
            inp = torch.zeros(1, inf_channels, cfg.backbone.image_size, cfg.backbone.image_size)
            _ = self.backbone(inp)
            self._fmap_size = self._fmap.shape[2]
            out_channels = self._fmap.shape[1]

        self.conv1 = nn.Conv2d(out_channels, 1, kernel_size=(1, 1), stride=(1,1), bias=True)

    def forward(self, x):
        x = self.backbone(x)
        fmap = self.conv1(self._fmap)
        return {'fmaps': fmap, 'logits': x}

    def get_fmap_size(self):
        return self._fmap_size


@register_model('net_wfmap_v2')
class NetWFMapV2(nn.Module):
    def __init__(self, cfg):
        super(NetWFMapV2, self).__init__()

        self.backbone = get_backbone(cfg, 1, exportable=False)
        def get_fmap(self):
            def hook(model, input, output):
                self._fmap = output
            return hook

        current_module = self.backbone
        for atr in cfg.backbone.fmap_block_seq:
            try:
                current_module = current_module.__getattr__(atr)
            except:
                current_module = current_module.__getitem__(atr)
        current_module.register_forward_hook(get_fmap(self))
        
        if cfg.use_HSV: # HSV
            inf_channels = 6
        else:
            inf_channels = 3
        
        with torch.no_grad():
            inp = torch.zeros(1, inf_channels, cfg.backbone.image_size, cfg.backbone.image_size)
            _ = self.backbone(inp)
            self._fmap_size = self._fmap.shape[2]
            out_channels = self._fmap.shape[1]

        self.conv1 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=(1, 1), stride=(1,1), bias=True)
        self.batchnorm1 = nn.BatchNorm2d(out_channels * 2)
        self.act1 = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 1), stride=(1,1), bias=True)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        x = self.backbone(x)
        fmap = self.act1(self.batchnorm1(self.conv1(self._fmap)))
        fmap = self.act2(self.batchnorm2(self.conv2(fmap)))
        fmap = self.conv3(fmap)
        return {'fmaps': fmap, 'logits': x}

    def get_fmap_size(self):
        return self._fmap_size
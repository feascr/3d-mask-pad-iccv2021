import torch
import torch.nn as nn
from . import create_getter


def build_HSV_conv(conv):
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        padding_mode = conv.padding_mode
        bias = conv.bias
        dilation = conv.dilation
        groups = conv.groups
        new_conv = nn.Conv2d(
            in_channels=6, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            padding_mode=padding_mode,
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        nn.init.kaiming_normal_(
            new_conv.weight, 
            nonlinearity='relu'
        )
        if bias:
            nn.init.constant_(new_conv.bias, 0)
        with torch.no_grad():
            weight = conv.weight.clone()
            new_conv.weight[:, :3] = weight
            # init HSV weights with RGB pretrained
            new_conv.weight[:, 3:6] = weight
        return new_conv

def get_backbone(cfg, num_classes, exportable):
    backbone_getter = create_getter(
        cfg.backbone.name,
        cfg.backbone.model_type,
        (cfg.backbone.pretrained and (cfg.model.pretrained_path is None)),
        exportable,
        num_classes,
        cfg.model.head_drop_rate,
        cfg.use_HSV,
        pretrained_path=cfg.backbone.pretrained_path
    )
    return backbone_getter.get_backbone()

def get_feature_extractor(cfg, exportable):
    backbone_getter = create_getter(
        cfg.backbone.name,
        cfg.backbone.model_type,
        (cfg.backbone.pretrained and (cfg.model.pretrained_path is None)),
        exportable,
        1,
        cfg.model.head_drop_rate,
        cfg.use_HSV,
        pretrained_path=cfg.backbone.pretrained_path
    )
    feature_extractor, out_size = backbone_getter.get_feature_extractor(
        image_size=cfg.backbone.image_size,
        use_bn_af=cfg.embedding_params.use_bn_af)
    return feature_extractor, out_size

def get_modules(cfg, num_classes, exportable):
    backbone_getter = create_getter(
        cfg.backbone.name, 
        cfg.backbone.model_type,
        (cfg.backbone.pretrained and (cfg.model.pretrained_path is None)),
        exportable,
        num_classes,
        cfg.model.head_drop_rate,
        cfg.use_HSV,
        pretrained_path=cfg.backbone.pretrained_path
    )
    feature_extractor, out_channels, out_size, fmap_size = backbone_getter.get_feature_extractor_from_block(
        cfg.backbone.fmap_block_seq, 
        image_size=cfg.backbone.image_size
    )
    fmap_conv = backbone_getter.build_feature_map_conv(out_channels)
    head = backbone_getter.build_fmap_head(out_size)
    return feature_extractor, fmap_conv, head, fmap_size
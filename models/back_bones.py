import torch
import torch.nn as nn
import timm
import logging
from . import register_getter


class BaseGetter:
    def __init__(self, model, num_classes=1, head_drop_rate=0., use_HSV=False):
        self.num_classes = num_classes
        self.model = model
        self.use_HSV = use_HSV
        self.head_drop_rate = head_drop_rate
        if self.use_HSV:
            self._add_HSV_to_model()
    
    @staticmethod
    def _init_weights(module, nonlinearity='relu', conv_bias=False):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
            if conv_bias:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.constant_(module.bias, 0)

    @staticmethod
    def _load_pretrained(model, pretrained_path):
        logging.debug('Loading pretrained weights for backbone')
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        missing_from_state_dict = {k: v for k, v in model.state_dict().items() if k not in state_dict}
        state_dict.update(missing_from_state_dict)
        model.load_state_dict(state_dict)
        return model


    def _build_HSV_conv(self, conv):
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        padding_mode = conv.padding_mode
        bias = conv.bias
        dilation = conv.dilation
        groups = conv.groups
        new_conv = nn.Conv2d(6, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
                                padding_mode=padding_mode,dilation=dilation, groups=groups, bias=bias)
        self._init_weights(new_conv, nonlinearity='relu', conv_bias=bias)
        with torch.no_grad():
            weight = conv.weight.clone()
            new_conv.weight[:, :3] = weight
            # init HSV weights with RGB pretrained
            new_conv.weight[:, 3:6] = weight
        return new_conv
    
    def build_fmap_head(self):
        head = nn.Sequential(
            [
                nn.Dropout(self.head_drop_rate), 
                nn.Sigmoid(), 
                nn.Flatten(), 
                nn.Linear(self.num_classes, 1)
            ]
        )
        self._init_weights(list(head.children())[-1], nonlinearity='sigmoid')
        return head
    
    def build_feature_map_conv(self, out_channels):
        fmap_conv = nn.Conv2d(out_channels, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self._init_weights(fmap_conv, nonlinearity='sigmoid', conv_bias=True)
        return fmap_conv    
    
    def _add_HSV_to_model(self):
        raise NotImplementedError


@register_getter('efficientnet')
class EfficientNetGetter(BaseGetter):
    def __init__(self, model_type, pretrained, exportable, num_classes=1, head_drop_rate=0., use_HSV=False, pretrained_path=None):
        assert model_type in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], 'EfficientNet model type is not supported'
        model_name = 'efficientnet_' + model_type
        model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            exportable=exportable, 
            num_classes=num_classes,
            drop_rate=head_drop_rate
        )
        if pretrained_path is not None:
            self._load_pretrained(model, pretrained_path)
        model = model.as_sequential()
        self.model_type = model_type
        super(EfficientNetGetter, self).__init__(model, num_classes, head_drop_rate, use_HSV)
    
    def _add_HSV_to_model(self):
        self.model[0] = self._build_HSV_conv(self.model[0])
    
    def get_backbone(self):
        return self.model

    def get_feature_extractor(self, image_size=224, use_bn_af=False, add_blocks=True):
        modules_list = []
        last_fe_block_seq = self._get_last_feature_extractor_block_seq(use_bn_af)
        
        named_children_generator = self.model.named_children()
        last_module = self.model
        for last_block in last_fe_block_seq:
            for name, module in named_children_generator:
                if name == last_block:
                    break
                modules_list.append(module)
            last_module = last_module.__getattr__(last_block)
            named_children_generator = last_module.named_children()
        if add_blocks:
            modules_list.extend(self._get_feature_extractor_additional_blocks(use_bn_af))
        feature_extractor = nn.Sequential(*modules_list)
        if self.use_HSV: # HSV
            inf_channels = 6
        else:
            inf_channels = 3

        with torch.no_grad():
            inp = torch.zeros(1, inf_channels, image_size, image_size)
            out_size = feature_extractor(inp).shape[1]
        
        return feature_extractor, out_size

    def get_feature_extractor_from_block(self, last_block_sequence, image_size=224):
        modules_list = []
        last_block = last_block_sequence[-1]
        named_children_generator = self.model.named_children()
        for name, module in named_children_generator:
            modules_list.append(module)
            if last_block == name:
                break
        
        feature_extractor = nn.Sequential(*modules_list)
        if self.use_HSV: # HSV
            inf_channels = 6
        else:
            inf_channels = 3

        with torch.no_grad():
            inp = torch.zeros(1, inf_channels, image_size, image_size)
            out_shape = feature_extractor(inp).shape
            out_channels = out_shape[1]
            out_size = out_shape[2] * out_shape[3]
            fmap_size = out_shape[2]
        
        return feature_extractor, out_channels, out_size, fmap_size

    @staticmethod
    def _get_last_feature_extractor_block_seq(use_bn_af=False):
        if use_bn_af:
            return ['15']
        else:
            return ['11']
    
    @staticmethod
    def _get_feature_extractor_additional_blocks(use_bn_af=False):
        if use_bn_af:
            return []
        else:
            return [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
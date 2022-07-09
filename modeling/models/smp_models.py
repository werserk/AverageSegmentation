import torch
from torch import nn
import segmentation_models_pytorch as smp
from ...utils import is_key_in_dict


class Unet(nn.Module):
    def __init__(self, cfg):
        super(Unet, self).__init__()
        self.cfg = cfg
        self.model = smp.Unet(cfg.backbone, classes=cfg.num_classes,
                              activation='softmax' if cfg.num_classes > 1 else 'sigmoid',
                              in_channels=cfg.in_channels)

        for i, x in enumerate(self.model.encoder.children()):
            if isinstance(x, torch.nn.Sequential):
                if cfg.layers_to_freeze:
                    for param in x.parameters():
                        param.requires_grad = False
                    cfg.layers_to_freeze -= 1

    def forward(self, x):
        return self.model(x)


class MAnet(nn.Module):
    def __init__(self, cfg):
        super(MAnet, self).__init__()
        self.cfg = cfg
        if 'resnext' not in cfg.backbone:
            self.model = smp.MAnet(cfg.backbone, classes=cfg.num_classes,
                                   activation='softmax' if cfg.num_classes > 1 else 'sigmoid',
                                   in_channels=cfg.in_channels)
        else:
            self.model = smp.MAnet(cfg.backbone, classes=cfg.num_classes,
                                   activation='softmax' if cfg.num_classes > 1 else 'sigmoid',
                                   in_channels=cfg.in_channels, encoder_weights=cfg.encoder_weights)
        for i, x in enumerate(self.model.encoder.children()):
            if isinstance(x, torch.nn.Sequential):
                if cfg.layers_to_freeze:
                    for param in x.parameters():
                        param.requires_grad = False
                    cfg.layers_to_freeze -= 1

    def forward(self, x):
        return self.model(x)


class UnetPlusPlus(nn.Module):
    def __init__(self, **kwargs):
        super(UnetPlusPlus, self).__init__()
        backbone = kwargs['backbone'] if is_key_in_dict(kwargs, 'backbone') else None
        num_classes = kwargs['num_classes'] if is_key_in_dict(kwargs, 'num_classes') else None
        in_channels = kwargs['in_channels'] if is_key_in_dict(kwargs, 'in_channels') else None
        layers_to_freeze = kwargs['layers_to_freeze'] if is_key_in_dict(kwargs, 'layers_to_freeze') else None
        encoder_weights = kwargs['encoder_weights'] if is_key_in_dict(kwargs, 'encoder_weights') else None
        if 'resnext' not in backbone:
            self.model = smp.UnetPlusPlus(backbone, classes=num_classes,
                                          activation='softmax' if num_classes > 1 else 'sigmoid',
                                          in_channels=in_channels)
        else:
            self.model = smp.UnetPlusPlus(backbone, classes=num_classes,
                                          activation='softmax' if num_classes > 1 else 'sigmoid',
                                          in_channels=in_channels, encoder_weights=encoder_weights)
        for i, x in enumerate(self.model.encoder.children()):
            if isinstance(x, torch.nn.Sequential):
                if layers_to_freeze:
                    for param in x.parameters():
                        param.requires_grad = False
                    layers_to_freeze -= 1

    def forward(self, x):
        return self.model(x)


class DeepLabV3(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3, self).__init__()
        self.cfg = cfg
        self.model = smp.DeepLabV3(cfg.backbone, classes=cfg.num_classes,
                                   activation='softmax' if cfg.num_classes > 1 else 'sigmoid',
                                   in_channels=cfg.in_channels)

        for i, x in enumerate(self.model.encoder.children()):
            if isinstance(x, torch.nn.Sequential):
                if cfg.layers_to_freeze:
                    for param in x.parameters():
                        param.requires_grad = False
                    cfg.layers_to_freeze -= 1

    def forward(self, x):
        return self.model(x)

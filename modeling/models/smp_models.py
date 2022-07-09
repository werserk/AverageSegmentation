import torch
from torch import nn
import segmentation_models_pytorch as smp


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
    def __init__(self, cfg):
        super(UnetPlusPlus, self).__init__()
        self.cfg = cfg
        if 'resnext' not in cfg.backbone:
            self.model = smp.UnetPlusPlus(cfg.backbone, classes=cfg.num_classes,
                                          activation='softmax' if cfg.num_classes > 1 else 'sigmoid',
                                          in_channels=cfg.in_channels)
        else:
            self.model = smp.UnetPlusPlus(cfg.backbone, classes=cfg.num_classes,
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

import sys
import torch


def get_setup(cfg, epochs, steps_per_epoch):
    model = _get_model(cfg)(**cfg.model_params).to(torch.device(cfg.device))
    optimizer = _get_optimizer(cfg)(model.parameters(), **cfg.optimizer_params)
    scheduler = _get_scheduler(cfg)(optimizer, **cfg.scheduler_params,
                                    epochs=epochs, steps_per_epoch=steps_per_epoch)
    return model, optimizer, scheduler


def get_metric(cfg):
    return _get_metric(cfg)(**cfg.metric_params)


def get_criterion(cfg):
    return _get_criterion(cfg)(**cfg.criterion_params)


def _get_metric(cfg):
    return getattr(sys.modules[cfg.metric_module], cfg.metric)


def _get_criterion(cfg):
    return getattr(sys.modules[cfg.criterion_module], cfg.criterion)


def _get_model(cfg):
    return getattr(sys.modules[cfg.models_module], cfg.model)


def _get_optimizer(cfg):
    return getattr(sys.modules[cfg.optimizer_module], cfg.optimizer)


def _get_scheduler(cfg):
    if cfg.scheduler:
        return getattr(sys.modules[cfg.scheduler_module], cfg.scheduler)
    return FakeScheduler


class FakeScheduler:
    def __init__(self, *args, **kwargs):
        self.lr = kwargs['lr']

    def step(self, *args, **kwargs):
        pass

    def get_last_lr(self, *args, **kwargs):
        return [self.lr]

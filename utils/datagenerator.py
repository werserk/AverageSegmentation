import os
import random
import albumentations as A
from torch.utils.data import DataLoader

from .dataset import CustomDataset


def split_paths(cfg, paths):
    _len = len(paths)
    splitted_paths = []
    last_size = 0
    assert sum(cfg.split_sizes) == 1 or sum(cfg.split_sizes) == _len, \
        f'Split sizes give summary {sum(cfg.split_sizes)} but have to give 1 or length of paths'
    for size in cfg.split_sizes:
        splitted_paths.append(paths[last_size:last_size + int(_len * size)])
        last_size += int(_len * size)
    return splitted_paths


def get_paths(cfg):
    paths = [os.path.join(cfg.data_folder, path) for path in os.listdir(cfg.data_folder)]
    random.shuffle(paths)
    return split_paths(cfg, paths)


def get_transforms(cfg):
    train_transforms = A.Compose([getattr(A, item["name"])(**item["params"]) for item in cfg.train_transforms])
    val_transforms = A.Compose([getattr(A, item["name"])(**item["params"]) for item in cfg.val_transforms])
    if cfg.test_transforms:
        test_transforms = A.Compose([getattr(A, item["name"])(**item["params"]) for item in cfg.test_transforms])
        return train_transforms, val_transforms, test_transforms
    return train_transforms, val_transforms


def get_loaders(cfg):
    paths = get_paths(cfg)
    transforms = get_transforms(cfg)

    train_ds = CustomDataset(paths[0], transform=transforms[0])
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, drop_last=True, shuffle=True)

    val_ds = CustomDataset(paths[1], transform=transforms[1])
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, drop_last=False)

    if len(paths) == 3:
        test_ds = CustomDataset(paths[2], transform=transforms[2])
        test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, drop_last=False)
        return train_dl, val_dl, test_dl

    return train_dl, val_dl

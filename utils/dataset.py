import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random


def read(paths):
    if isinstance(paths, str):
        image_path, mask_path = paths, None
    elif isinstance(paths, tuple):
        image_path, mask_path = paths
    elif isinstance(paths, list):
        image_path, mask_path = paths
    elif isinstance(paths, np.ndarray):
        image_path, mask_path = paths
    else:
        raise TypeError(f'Unsupported type "{type(paths)}" for loading images')

    if any([image_path.endswith(x) for x in ('.jpg', '.png', '.jpeg')]):  # read in rgb
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mask_path:
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = np.expand_dims(mask, axis=-1)
        else:
            mask = np.zeros((image.shape[0], image.shape[1], 1))
    elif image_path.endswith('.npz'):
        sample = np.load(image_path)
        image = sample['image']
        mask = sample['mask']
    else:
        raise TypeError(f'Unsupported image type "{image_path}" for loading images')
    return image, mask


class CustomDataset(Dataset):
    def __init__(self, paths, transform=None, classes=None):
        if classes is None:
            classes = []
        self.paths = paths
        self.transform = transform
        self._len = len(self.paths)
        self.classes = classes

    def get_transforms(self, *args, **kwargs):
        return self.transform

    def get_classes(self, *args, **kwargs):
        return self.classes

    def __len__(self):
        return self._len

    def __add__(self, dataset):
        if isinstance(dataset, CustomDataset):
            return MultipleDataset([self.paths, dataset.paths],
                                   [self.transform, dataset.transform],
                                   [self.classes, dataset.classes])
        elif isinstance(dataset, MultipleDataset):
            return MultipleDataset([self.paths, *dataset.original_paths],
                                   [self.transform, *dataset.original_transforms],
                                   [self.classes, *dataset.original_classes])
        else:
            raise TypeError(f'Unsupported dataset type "{type(dataset)}"')

    def __getitem__(self, index):
        path = self.paths[index]
        image, mask = read(path)
        transform = self.get_transforms(path)
        classes = self.get_classes(path)

        # using transform if needed
        if transform:
            transformed = transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # to [0..1]
        if image.max() > 1:
            image = image / 255

        # if all labels in 1 dimension, we split them to number of labels
        if len(classes) > 2 and mask.shape[2] == 1:
            new_mask = [(mask == label)[:, :, 0] for label in classes]
            new_mask = np.array(new_mask)
            mask = np.transpose(new_mask, (1, 2, 0))

        # data type and dimension correction
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(np.array(image, dtype=np.float))
        image = image.type(torch.FloatTensor)
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        return image, mask


class MultipleDataset(CustomDataset):
    def __init__(self, paths, transforms, classes):
        super(MultipleDataset, self).__init__(paths)
        self.original_paths = paths
        self.original_transforms = transforms
        self.original_classes = classes
        self.paths2ind = {}
        self.paths = list(self.paths2ind.keys())
        for i, _paths in enumerate(paths):
            for path in _paths:
                self.paths2ind[path] = i
        random.shuffle(paths)

    def get_transforms(self, path):
        return self.original_transforms[self.paths2ind[path]]

    def get_classes(self, path):
        return self.original_classes[self.paths2ind[path]]

    def __add__(self, dataset):
        if isinstance(dataset, CustomDataset):
            return MultipleDataset([*self.original_paths, dataset.paths],
                                   [*self.original_transforms, dataset.transform],
                                   [*self.original_classes, dataset.classes])
        elif isinstance(dataset, MultipleDataset):
            return MultipleDataset([*self.original_paths, *dataset.original_paths],
                                   [*self.original_transforms, *dataset.original_transforms],
                                   [*self.original_classes, *dataset.original_classes])
        else:
            raise TypeError(f'Unsupported dataset type "{type(dataset)}"')

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from ..utils import is_key_in_dict


def minmax_normalize(image):
    _max = image.max()
    _min = image.min()
    if _max == _min:
        return image
    image = (image - _min) / (_max - _min)
    return image


class Visualizer:
    def __init__(self, **kwargs):
        self.figsize = (6, 6)
        self.axis = False
        self.normalize_function = minmax_normalize
        self.combined = False
        self.alpha = 0.25
        self.__dict__.update(kwargs)

    def imshow(self, image, mask_pred=None, mask_true=None, **kwargs):
        checkpoint = self.__dict__  # remember global setup
        self.__dict__.update(kwargs)  # update them to call this function with custom setup

        # configure plot
        if mask_pred and mask_true:
            fig, axes = plt.subplots(1, 2)
            fig.set_figwidth(self.figsize[0] * 2)
        else:
            fig, axes = plt.subplots(1, 1)
            fig.set_figwidth(self.figsize[0])
        fig.set_figheight(self.figsize[1])

        # (not) show axis
        if not self.axis:
            for ax in axes:
                ax.set_axis_off()

        # visualize image
        image = self.normalize_function(image)
        axes[0].imshow(image)

        # visualize mask
        if mask_pred is not None:
            axes[0].imshow(mask_pred, alpha=self.alpha)
        if mask_true is not None:
            axes[1].imshow(image)
            axes[1].imshow(mask_true, alpha=self.alpha)
        plt.show()
        self.__dict__ = checkpoint  # come back to previous setup

    def to_numpy(self, a):
        if isinstance(a, str):
            bgr_img = cv2.imread(a)
            assert bgr_img is not None, f"can't read '{a}'"
            a = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        elif isinstance(a, torch.Tensor) and len(a.shape) == 3:
            a = np.array(a.detach().cpu())
            a = np.transpose(a, (1, 2, 0))
        elif not (isinstance(a, np.ndarray) and len(a.shape) == 3):
            raise TypeError(f"Unknown type '{type(a)}' or not correct size '{a.shape}'")
        return a

    def __call__(self, a, **kwargs):
        if isinstance(a, torch.utils.data.DataLoader):
            image_number = kwargs['image_number'] if is_key_in_dict(kwargs, 'image_number') else -1
            k = 0
            for batch in a:
                batch = zip(*batch)
                for image, mask in batch:
                    image = self.to_numpy(image)
                    mask = self.to_numpy(mask)
                    self.imshow(image, mask, **kwargs)
                    k += 1
                    if k == image_number:
                        return
        if len(a) == 2:
            image, mask = a
            image = self.to_numpy(image)
            mask = self.to_numpy(mask)
            self.imshow(image, mask, **kwargs)
        elif len(a) == 3:
            image, mask_pred, mask_true = a
            image = self.to_numpy(image)
            mask_pred = self.to_numpy(mask_pred)
            mask_true = self.to_numpy(mask_true)
            self.imshow(image, mask_pred, mask_true, **kwargs)
        else:
            image = a
            image = self.to_numpy(image)
            self.imshow(image, **kwargs)

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
    figsize = (6, 12)
    axis = False
    normalize_function = minmax_normalize
    combined = False
    overlap_percents = 100

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def imshow(self, image, mask=None, **kwargs):
        checkpoint = self.__dict__  # remember global setup
        self.__dict__.update(kwargs)  # update them to call this function with custom setup

        # configure plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(self.figsize[1])
        fig.set_figwidth(self.figsize[0])

        # (not) show axis
        if not self.axis:
            ax1.set_axis_off()
            ax2.set_axis_off()

        # visualize image
        image = self.normalize_function(image)
        ax1.imshow(image)

        # visualize mask
        if mask is not None:
            # combine mask and image if needed
            if self.combined:
                filtered_image = image * (1 - mask * self.overlap_percents / 100)
                mask = mask * (image.max() * self.overlap_percents / 100)
                mask = mask + filtered_image
                mask = self.normalize_function(mask)
            ax2.imshow(mask)
        plt.show()
        self.__dict__ = checkpoint  # come back to previous setup

    def to_numpy(self, a):
        if isinstance(a, str):
            bgr_img = cv2.imread(a)
            assert bgr_img is not None, f"can't read '{a}'"
            a = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        elif isinstance(a, torch.Tensor) and len(a.shape) == 3:
            a = np.array(a)
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
        else:
            image = a
            image = self.to_numpy(image)
            self.imshow(image, **kwargs)

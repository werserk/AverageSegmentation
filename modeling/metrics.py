import torch


class IoUScore:
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def __call__(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, str(y_pred.shape) + ' != ' + str(y_true.shape)
        intersection = torch.sum(y_true * y_pred, dim=[2, 3])
        total = torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3])
        union = total - intersection
        iou = ((intersection + self.eps) / (union + self.eps)).mean(dim=1).mean(dim=0)
        return 1 - iou


class DiceScore:
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def __call__(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, str(y_pred.shape) + ' != ' + str(y_true.shape)
        intersection = torch.sum(y_true * y_pred, dim=[2, 3])
        total = torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3])
        dice = ((intersection + self.eps) / (total + self.eps)).mean(dim=1).mean(dim=0)
        return 1 - dice

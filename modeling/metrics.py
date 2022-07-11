class IoUScore(object):
    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, inputs, targets):
        assert inputs.shape == targets.shape, f"{inputs.shape} - {targets.shape}"
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + self.smooth) / (union + self.smooth)
        return IoU


class DiceLoss(object):
    def __init__(self, smooth):
        self.smooth = smooth

    def forward(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

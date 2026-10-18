
import torch
import torch.nn.functional as F


class MeanIoU(object):
    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # row：GT；pred：Pred
        self.confusion = torch.zeros(
            num_classes, num_classes, dtype=torch.int64
        )

    @torch.no_grad()
    def update(self, preds, target):
        """
        preds:  [B, C, H, W] logits 或 [B, H, W] 类别预测
        target: [B, H, W] GT 类别索引
        """
        if preds.ndim == target.ndim + 1:
            preds = preds.argmax(dim=1)

        preds = preds.reshape(-1).long()
        target = target.reshape(-1).long()

        # ignore invalid target
        valid = (
            (target >= 0)
            & (target < self.num_classes)
            & (preds >= 0)
            & (preds < self.num_classes)
        )

        if self.ignore_index is not None:
            valid &= target != self.ignore_index

        preds = preds[valid]
        target = target[valid]

        # flatten 2D confusion matrix indices into 1D indices
        indices = target * self.num_classes + preds

        batch_confusion = torch.bincount(
            indices,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion += batch_confusion.cpu()

    def compute(self):
        """
        返回：
            miou: 标量
            iou:  [C]，每个类别的 IoU
        """
        confusion = self.confusion.float()

        tp = confusion.diag()
        fp = confusion.sum(dim=0) - tp
        fn = confusion.sum(dim=1) - tp

        union = tp + fp + fn

        # classes absent from both the ground truth
        # and predictions are excluded from the average
        valid_classes = union > 0

        iou = torch.full(
            (self.num_classes,), float("nan")
        )
        iou[valid_classes] = (
            tp[valid_classes] / union[valid_classes]
        )

        if valid_classes.any():
            miou = iou[valid_classes].mean()
        else:
            miou = torch.tensor(float("nan"))

        return miou.item(), iou

    def reset(self):
        self.confusion.zero_()


def mean_iou(preds, target, num_classes, ignore_index=None, smooth=1e-10):
    """

    Args:
        preds: [B, C, H, W]
        target: [B, H, W]
        num_classes:
        ignore_idx:

    Returns:

    """
    # flatten
    preds = preds.argmax(dim=1).reshape(-1)
    target = target.reshape(-1)

    # valid
    if ignore_index is not None:
        valid = target != ignore_index
        preds = preds[valid]
        target = target[valid]

    # state confusion matrix
    indices = target * num_classes + preds
    confusion = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)

    # cal tp, fp, fn
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    # cal iou
    # intersection / union
    union = tp + fp + fn
    iou = tp / (union + smooth)
    miou = iou.mean()

    return miou, iou


if __name__ == "__main__":
    B, C, H, W = 2, 4, 32, 32
    logits = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))

    miou, iou = mean_iou(
        logits, target, num_classes=C, ignore_index=255
    )

    print("Per-class IoU:", iou)
    print("mIoU:", miou.item())



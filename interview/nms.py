import numpy as np
import torch
from torch import Tensor
import torchvision
from interview.iou import box_iou


def nms(bboxes: torch.Tensor, scores: Tensor, iou_threshold: float):
    """

    Args:
        bboxes:
        scores:
        iou_threshold:

    Returns:
        Tensor: the indices of the elements that have been kept by nms
    """

    assert bboxes.shape[0] == scores.shape[0]

    # descend order by score
    _, order = torch.sort(scores, descending=True)

    # record keep index
    keep = []

    # num
    while order.numel() > 0:
        i = order[0].item()
        # reserve index of the current highest score
        keep.append(i)

        # stop condition
        if order.numel() == 1:
            break

        # cal iou between the highest-score box and the remaining box
        iou = box_iou(bboxes[i][None, :], bboxes[order[1:]])[0]

        # keep the box with iou less threshold
        mask = iou <= iou_threshold

        # update remaining indices
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long)


if __name__ == "__main__":
    bboxes = torch.tensor([
        [10, 10, 50, 50],
        [15, 15, 55, 55],
        [60, 60, 100, 100],
        [70, 70, 110, 110]
    ], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.5, 0.7, 0.6], dtype=torch.float32)
    iou_threshold = 0.5

    keep_index = nms(bboxes, scores, iou_threshold)
    keep_bboxes = bboxes[keep_index]
    print(keep_bboxes)



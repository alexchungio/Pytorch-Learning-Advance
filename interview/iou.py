
import torch


def compute_iou(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """
    Calculate overlap between two set of bboxes.
    Args:
        bboxes1: (M, 4) <x1, y1, x2, y2>
        bboxes2: (N, 4) <x1, y1, x2, y2>
        mode: 'iou(intersection over union)' or 'giou(generalized intersection over union)'

    Returns:
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".

        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (N, M)

    """

    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * ( bboxes1[..., 3] - bboxes1[..., 1])  # (M, )
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * ( bboxes2[..., 3] - bboxes2[..., 1])  # (N, )

    # compute inter
    # left-top
    lt = torch.max(bboxes1[:, None, :2],
                   bboxes2[None, :, :2])  # [M, N, 2]
    # right-bottom
    rb = torch.min(bboxes1[:, None, 2:],
                   bboxes2[None, :, 2:])  # [M, N 2]
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)

    overlap = wh[..., 0] * wh[..., 1]  # (N, M)

    union = area1[:, None] + area2[None, :] - overlap  # (M, N)

    ious = overlap / (union + eps)

    if mode == 'giou':
        enclosed_lt = torch.min(bboxes1[:, None, :2],
                                bboxes2[None, :, :2])
        enclosed_rb = torch.max(bboxes1[:, None, 2:],
                                bboxes2[None, :, 2:])

        # calculate gious
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]  # (M, N)
        gious = ious - (enclose_area - union) / (enclose_area + eps)

        return gious

    return ious


if __name__ == "__main__":

    bboxes1 = torch.FloatTensor([[0, 0, 10, 10],
                                [10, 10, 20, 20],
                                [15, 15, 30, 30]])

    bboxes2 = torch.FloatTensor([[0, 0, 10, 20],
                                [10, 20, 20, 30]])

    iou = compute_iou(bboxes1, bboxes2, mode='iou')
    print(iou)
    giou = compute_iou(bboxes1, bboxes2, mode='giou')
    print(giou)






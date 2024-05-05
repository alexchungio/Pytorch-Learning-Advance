import os

import torch


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        """
        computer confusion matrix of per sample and update
        Args:
            a: (1-d tensor): target - GT, dtype=torch.int64
            b: (1-d tensor): output.argmax(1), - pred, dtype=torch.int64

        Returns:

        num_classes = 3
        target =>  [0, 0, 1, 2, 2, 1, 2, 1, 1]
        predict => [0, 0, 2, 2, 1, 1, 2, 1, 0]

        step 1: set label index of per class as raw start index at confusion matrix
                0 indicate class-0, 3 indicate class-1(3*1), 6 indicate class-2(3*2)
                [[0, 0, 0],  # class-0  [0][0] => 0*3+0 = 0
                 [0, 0, 0],  # class-1  [1][0] => 1*3+0 = 3
                 [0, 0, 0]]  # class-2  [2][0] => 2*3+0 = 6
                target * 3 => [0, 0, 3, 6, 6, 3, 6, 3, 3]
        step 2: get intdex of predict label  at confusion matrix
                [0, 0, 3, 6, 6, 3, 6, 3, 3] + [0, 0, 2, 2, 1, 1, 2, 1, 0]  => [0, 0, 5, 8, 7, 4, 8, 4, 3]
        step3: count the number of per index to generate confusion matrix
               [2, 1, 0, 1, 2, 1, 0, 1, 2] =>
               [[2, 1, 0],
                [1, 2, 1],
                [0, 1, 2]]

        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)  # get valid index
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        """
        computer miou with confusion matrix
        Returns:

        """
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()  # accuracy of all class
        acc = torch.diag(h) / h.sum(1)  # accuracy of per class
        iou = torch.diag(h) / (h.sum(0) + h.sum(1) - torch.diag(h))  # iou of per lcass

        return acc_global, acc, iou

    def metric(self):
        _, _, iou = self.compute()
        mean_iou = iou.mean().item() * 100

        return {"meanIoU": mean_iou}

    def __str__(self):
        acc_global, acc, iou = self.compute()
        return (
            'global accuracy: {:.2f} \n'
            'average row accuracy: {} \n'
            'IoU per class: {} \n'
            'meanIoU: {:.2f}').format(
                acc_global.item() * 100,
                ['{:.2f}'.format(a) for a in (acc * 100).tolist()],
                ['{:.2f}'.format(i) for i in (iou * 100).tolist()],
                iou.mean().item() * 100
            )


def main():
    torch.random.manual_seed(2023)

    confusion_matrix = ConfusionMatrix(num_classes=2)
    target = torch.randint(low=0, high=2, size=(3, 3))
    label = torch.randint(low=0, high=2, size=(3, 3))

    confusion_matrix.update(target, label)

    print(confusion_matrix)


if __name__ == "__main__":
    main()
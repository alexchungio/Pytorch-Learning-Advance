import torch
import torch.nn as nn


if __name__ == "__main__":
    x = torch.arange(1, 17, step=1, dtype=torch.float32)
    x = x.reshape((1, 1, 4, 4))

    # no overlap sliding =>  kernel_size <= stride
    unfold_0 = nn.Unfold(kernel_size=2, stride=2, padding=0)
    fold_0 = nn.Fold(output_size=(4, 4), kernel_size=2, stride=2, padding=0)
    patch_0 = unfold_0(x)
    x_0 = fold_0(patch_0)
    print(x)
    print(patch_0)
    print(x_0)

    # no overlap sliding =>  kernel_size <= stride
    print('+'*40)
    unfold_1 = nn.Unfold(kernel_size=2, stride=1, padding=0)
    fold_1 = nn.Fold(output_size=(4, 4), kernel_size=2, stride=1, padding=0)
    patch_1 = unfold_1(x)
    x_1 = fold_1(patch_1)

    print(patch_1)
    """
    [[[[ 1,  2+2,      3+3,     4],
       [5+5, 6+6+6+6., 7+7+7+7, 8+8],
       [9+9, 10+10+101+10, 11+11+11+11, 12+12.],
       [13,  14+14,    15+15,  16]]]] 
    = [[[[ 1,  4,  6,  4],
          [10, 24, 28, 16],
          [18, 40, 44, 24],
          [13, 28, 30, 16]]]]
    """
    print(x_1)
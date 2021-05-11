import torch


def gather_demo():
    """
    reference https://pytorch.org/docs/stable/generated/torch.gather.html
    Gathers values along an axis specified by dim.

    For a 3-D tensor the output is specified by:

    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    """
    t = torch.tensor([[1, 2], [3, 4]])
    index = torch.LongTensor([[0, 0], [1, 0]])
    gather_t0 = torch.gather(t, dim=1, index=index)
    """
        [[t[0][index[0][0]]]->t[0][0]=1, t[0][index[0][1]]]->t[0][0]=1]
         [t[1][index[1][0]]]->t[1][1]=4, t[1][index[1][1]]]->t[1][0]=3]]
        =
         [[1, 1],
          [4, 3]]
    """
    print(gather_t0)
    gather_t1 = torch.gather(t, dim=0, index=index)
    """
        [[t[index[0][0]]][0]->t[0][0]=1, t[index[0][1]]][1]->t[0][1]=2]
         [t[index[1][0]]][0]->t[1][0]=3, t[index[1][1]]][1]->t[0][1]=2]]
        =
         [[1, 2],
          [3, 2]]
    """
    print(gather_t1)


def scatter_demo():
    """
    reference https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_
    Writes all values from the tensor src into self at the indices specified in the index tensor. For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.

    For a 3-D tensor, self is updated as:

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
    This is the reverse operation of the manner described in gather().
    """

    src = torch.arange(1, 11).reshape((2, 5))
    print(src)
    '''
    tensor([[ 1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10]])
    '''
    dim_0 = 0
    index_0 = torch.tensor([[0, 1, 2, 0]])   # !!! max([0, 1, 2, 0], dim=dim_0) < src.shape[dim_0]
    scatter_0 = torch.zeros(3, 5, dtype=src.dtype).scatter_(dim=dim_0, index=index_0, src=src)
    print(scatter_0)

    '''
    [index[0][0], 0] = [0, 0] = src[0, 0] = 1
    [index[0][1], 1] = [1, 1] = src[0, 1] = 2
    [index[0][2], 2] = [2, 2] = src[0, 2] = 3
    [index[0][3], 3] = [0, 3] = src[0, 3] = 4
    ...
    '''

    dim_1 = 0
    index_1 = torch.tensor([[0, 1, 2], [0, 1, 4]])
    scatter_1 = torch.zeros(3, 5, dtype=src.dtype).scatter_(dim=dim_1, index=index_1, src=src)
    print(scatter_1)
    '''
    [0, index[0, 0]] = [0, 0] = src[0, 0] = 1
    [0, index[0, 1]] = [0, 1] = src[0, 1] = 2
    [0, index[0, 2]] = [0, 2] = src[0, 2] = 3
    [1, index[1, 0]] = [1, 0] = src[1, 0] = 6
    [1, index[1, 1]] = [1, 1] = src[1, 1] = 7
    [1, index[1, 2]] = [1, 4] = src[1, 2] = 8
    '''

    scatter_2 = torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]), 1.23, reduce='multiply')
    print(scatter_2)

    scatter_3 = torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]), 1.23, reduce='add')
    print(scatter_3)


def main():
    gather_demo()
    scatter_demo()


if __name__ == "__main__":
    main()
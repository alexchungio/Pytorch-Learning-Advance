import torch

if __name__ == "__main__":

    tensor = torch.tensor([1, 3, 2, 4, 1, 2, 5, 4, 3, 1])
    unique_number = torch.unique(tensor)
    print(unique_number)  # tensor([1, 2, 3, 4, 5])

    unique_number, number_index = torch.unique(tensor, return_inverse=True)
    new_tensor = unique_number[number_index]
    print(unique_number)  # tensor([1, 2, 3, 4, 5])
    print(number_index)  # tensor([0, 1, 2, 1, 0, 3, 4, 3, 2, 0])
    assert torch.equal(tensor, new_tensor)

    unique_number, number_count = torch.unique(tensor, return_counts=True)
    print(unique_number)  # tensor([1, 2, 3, 4, 5])
    print(number_count)  # tensor([3, 2, 2, 2, 1])

    # unique_number, number_index, number_count = torch.unique(tensor, return_inverse=True, return_counts=True)

    unique_number, number_index, number_count = torch.unique(tensor, sorted=True,
                                                             return_inverse=True, return_counts=True)
    print(unique_number)  # tensor([1, 2, 3, 4, 5])
    print(number_index)  # tensor([0, 2, 1, 3, 0, 1, 4, 3, 2, 0])
    print(number_count)  # tensor([3, 2, 2, 2, 1])




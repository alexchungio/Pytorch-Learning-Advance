import torch
import numpy as np


def custom_cosine_similar(input1, input2):
    """

    Args:
        input1:
        input2:

    Returns:

    """
    dot = torch.sum(torch.multiply(input1, input2), dim=1)
    norm_1 = torch.norm(input1, p='fro', dim=1)
    norm_2 = torch.norm(input2, p='fro', dim=1)

    cosine = dot / (norm_1 * norm_2)

    return cosine


def custom_cosine_embedding_loss(input1, input2, target, margin=0):
    """

    Args:
        input1:
        input2:
        target:
        margin:

    Returns:

    """
    # calculate cosine

    cosine_similar = custom_cosine_similar(input1, input2)

    # calculate loss of target == 1
    cosine_similar_0 = cosine_similar[target == 1]
    loss_0 = 1 - cosine_similar_0

    # calculate loss of target == -1
    cosine_similar_1 = cosine_similar[target == -1]
    loss_1 = torch.clamp_min(cosine_similar_1 - margin, min=0)

    # reduce mean
    loss = torch.mean(torch.cat([loss_0, loss_1], dim=0))

    return loss


def main():
    np.random.seed(2021)
    torch.random.manual_seed(2021)
    feature_0 = torch.randn(size=(4, 5))
    feature_1 = torch.randn(size=(4, 5))

    flag_0 = torch.Tensor(np.random.choice([-1, 1], size=feature_0.shape[0], p=(0, 1)))
    flag_1 = torch.Tensor(np.random.choice([-1, 1], size=feature_0.shape[0], p=(0.5, 0.5)))

    cosine_embedding = torch.nn.CosineEmbeddingLoss()
    cosine_similar = torch.nn.CosineSimilarity()

    # step 1 calculate loss with CosineEmbeddingLoss
    loss_0 = cosine_embedding(feature_0, feature_1, target=flag_0)
    loss_1 = cosine_embedding(feature_0, feature_1, target=flag_1)

    # step 2 calculate loss with cosine_simlar
    cosine_0 = cosine_similar(feature_0, feature_1)
    cosine_1 = custom_cosine_similar(feature_0, feature_1)
    # get cosine embedding loss
    loss_2 = 1 - torch.mean(cosine_0)

    # step 3 calculate loss with custom function
    loss_3 = custom_cosine_embedding_loss(feature_0, feature_1, target=flag_0)
    loss_4 = custom_cosine_embedding_loss(feature_0, feature_1, target=flag_1)

    # show result
    assert cosine_0.all() == cosine_1.all()
    assert loss_0 == loss_2
    assert loss_0 == loss_3
    assert loss_1 == loss_4
    print(flag_0)
    print(flag_1)

    print(loss_0)
    print(loss_1)



if __name__ == "__main__":
    main()

# Nonuniform-to-Uniform Quantization: Towards Accurate Quantization via Generalized Straight-Through Estimation
import torch

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)   # 前向真实量化

    @staticmethod
    def backward(ctx, grad_output):
        # return grad_output.clamp_(-1, 1)
        return grad_output


if __name__ == "__main__":
    # round_ste = RoundSTE.apply
    intput_data = torch.randn(4, requires_grad=True)
    output = RoundSTE.apply(intput_data)
    loss = output.mean()
    loss.backward()

    print(intput_data.grad)





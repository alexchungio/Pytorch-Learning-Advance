import torch
from torch import nn, Tensor
from typing import Dict, Iterable, Callable

"""
钩子（hook）指通过拦截软件模块间的函数调用、消息传递、事件传递来来修改或扩展应用程序或软件行为的各种技术。
处理被拦截的函数调用、事件、消息的代码，被成为钩子（hook）
观察者模式的内部机制可以通过hook机制实现

"""


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*4*4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feature = self.backbone(x)
        output = self.classifier(feature)

        return output


# define hook callback function
def module_forward_hook(module: nn.Module, input: Tensor, output:Tensor):
    pass


def module_backbone_hook(module: nn.Module, input_grad: Tensor, output_grad:Tensor):
    pass


def tensor_hook(grad: Tensor):
    pass


class VerboseModel(nn.Module):
    """
    show model verbose
    """
    def __init__(self, model:nn.Module, module_type=(nn.Conv2d, nn.Linear)):
        super().__init__()
        self.model = model
        self.module_type = module_type

        # register hook for each module
        # note!!! register before forward
        for name, module in self.model.named_modules():
            if isinstance(module, self.module_type):
                module.__name__ = name
                module.register_forward_hook(lambda module, _, output: print(f'{module.__name__}: {output.shape}'))

    def forward(self, x):

        return self.model(x)


class FeatureExtraction(nn.Module):
    """
    extract model feateure
    """
    def __init__(self, model: nn.Module, module_idx: Iterable[str]):
        super().__init__()
        self.model = model
        self.module_idx = module_idx
        self._feature = {}

        # register before
        for module_id in module_idx:
            module = dict([*self.model.named_modules()])[module_id] # get module
            module.register_forward_hook(self.save_output_hook(module_id))

    def save_output_hook(self, module_id) -> Callable:
        """

        Args:
            module_id:

        Returns:

        """
        def hook_fn(module, _, output):
            self._feature[module_id] = output

        return hook_fn

    def forward(self, x) -> [Dict, Tensor]:
        predict = self.model(x)

        return predict, self._feature


def gradient_clipper(model: nn.Module, val:float) -> nn.Module:
    """
    gradient clipper hook
    Args:
        model:
        val:

    Returns:

    """
    for param in model.parameters():
        param.register_hook(lambda grad: grad.clamp_(-val, val))
    return model


def main():

    torch.random.manual_seed(2022)
    num_classes = 10
    dummy_input = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    dummy_target = torch.tensor([2, 6], dtype=torch.long)
    dummy_feature = torch.rand(2, 16, 4, 4, dtype=torch.float32)

    model = Model(num_classes)
    out = model(dummy_input)
    
    # show children
    for name, _ in model.named_children():
        print(name)
    # show module
    for name, _ in model.named_modules():
        print(name)

    # test VerboseModel
    verbose_model = VerboseModel(model)
    _ = verbose_model(dummy_input)

    # test gradient_clipper
    criterion_0 = nn.CrossEntropyLoss()
    clip_model = gradient_clipper(model, 0.2)  # register hook
    predict = clip_model(dummy_input)
    loss = criterion_0(predict, dummy_target)
    loss.backward()
    print(clip_model.backbone[3].bias.grad)
    print(clip_model.classifier[4].bias.grad)

    # test FeatureExtraction
    features_extraction = FeatureExtraction(model, ['backbone.6', 'classifier.4'])
    predict, features = features_extraction(dummy_input)
    print({name: output.shape for name, output in features.items()})

    # test gradient backward of hook
    criterion_1 = nn.MSELoss()
    loss_0 = criterion_0(predict, dummy_target)
    loss_1 = criterion_1(features['backbone.6'], dummy_feature)  # use hook feature
    loss = loss_0 + loss_1
    loss.backward()


if __name__ == "__main__":
    main()

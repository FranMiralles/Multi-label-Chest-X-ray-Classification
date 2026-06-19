import torch.nn as nn
from torchvision import models


def build_densenet121(n_classes: int, pretrained: bool = True) -> nn.Module:
    '''
    Constructs a DenseNet-121 model, optionally loading pretrained ImageNet weights.
    The final classifier layer is replaced to output `n_classes` logit scores.
    Params:
        - n_classes: Number of output classes for the classifier layer.
        - pretrained: If True, loads ImageNet pretrained weights; otherwise, initializes randomly.
    Returns:
        - A PyTorch nn.Module representing the modified DenseNet-121 model.
    '''
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.densenet121(weights=weights)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, n_classes)
    return model

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

# Feature dimensions after global pooling (before classifier):
# ResNet18/34: 512, ResNet50/101: 2048

def _remove_fc(model):
    model.fc = nn.Identity()
    return model

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, projector=None, **kwargs):
        super().__init__()
        self.model = _remove_fc(resnet18(pretrained=pretrained, **kwargs))
        self.projector = projector if projector is not None else nn.Identity()
    def forward(self, x):
        features = self.model(x)  # shape: (B, 512)
        return self.projector(features)

class ResNet34(nn.Module):
    def __init__(self, pretrained=False, projector=None, **kwargs):
        super().__init__()
        self.model = _remove_fc(resnet34(pretrained=pretrained, **kwargs))
        self.projector = projector if projector is not None else nn.Identity()
    def forward(self, x):
        features = self.model(x)  # shape: (B, 512)
        return self.projector(features)

class ResNet50(nn.Module):
    def __init__(self, pretrained=False, projector=None, **kwargs):
        super().__init__()
        self.model = _remove_fc(resnet50(pretrained=pretrained, **kwargs))
        self.projector = projector if projector is not None else nn.Identity()
    def forward(self, x):
        features = self.model(x)  # shape: (B, 2048)
        return self.projector(features)

class ResNet101(nn.Module):
    def __init__(self, pretrained=False, projector=None, **kwargs):
        super().__init__()
        self.model = _remove_fc(resnet101(pretrained=pretrained, **kwargs))
        self.projector = projector if projector is not None else nn.Identity()
    def forward(self, x):
        features = self.model(x)  # shape: (B, 2048)
        return self.projector(features)

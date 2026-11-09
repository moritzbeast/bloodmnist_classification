import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class MyModel(torch.nn.Module):
    def __init__(self, backbone="resnet18"):
        super(MyModel,self).__init__()

        # load weights pretrained on ImageNet
        if backbone == "resnet18":
            self.model = models.resnet18(weights=None)

        elif backbone == "resnet18_modified":
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # stem for small images: 3x3 conv without stride and no maxpool,
            # so a 28x28 image ends as 4x4 instead of 1x1 after layer4
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()

        elif backbone == "resnet34":
            self.model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        elif backbone == "resnet50":
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # change the number of outputs according to our problem
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)

    def forward(self, x):
        return self.model(x)

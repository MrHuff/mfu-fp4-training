import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample 2x
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
class CNNModelCIFAR(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample 2x
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.LayerNorm(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8),
            nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class ImageNet100(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=False)


    def forward(self, x):
        return self.resnet18(x)

class LinearModel(torch.nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


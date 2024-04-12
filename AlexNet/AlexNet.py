import torch
from torchviz import make_dot
import torchinfo
import torch.nn as nn

torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicAlexNet(nn.Module):
    def __init__(self, num_classes: int = 200) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.relu = nn.ReLU(inplace=True)

        self.features = nn.Sequential(
            self.conv1, self.relu, self.pool1,
            self.conv2, self.relu, self.pool2,
            self.conv3, self.relu,
            self.conv4, self.relu,
            self.conv5, self.relu, self.pool5
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.flattener = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flattener(x)
        x = self.classifier(x)
        return x

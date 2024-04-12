import torch
import torch.nn as nn

# Define the Basic Block which is used in ResNet18 and ResNet34
# ResNet50 uses a "Bottleneck" block which is defined below


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer of the BasicBlock
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer of the BasicBlock
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample layer adjusts the size and dimension to match the output if needed
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Pass through the first conv-bn-relu layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)

        # Pass through the second conv-bn layers
        out = self.conv2(out)
        out = self.bn2(out)

        # If downsample is required (e.g., dimension mismatch), apply it to the residual
        if self.downsample is not None:
            residual = self.downsample(x)

        # Add the residual to the output and apply ReLU
        out += residual
        out = nn.ReLU()(out)

        return out

# Bottleneck block used in ResNet50/101/152


class Bottleneck(nn.Module):
    expansion = 4  # Expands the last layer of the block by a factor of 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # The first convolution reduces the channel size to out_channels from in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # The second convolution keeps the channel size constant but may reduce spatial dimensions
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # The third convolution increases the channel size by a factor of the expansion
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # Downsample layer if needed for dimension matching
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = nn.ReLU()(out)

        return out

# ResNet architecture


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.in_channels = 64
        super(ResNet, self).__init__()
        # Initial convolutional layer before entering the residual blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Creating the layers with residual blocks
        # Each layer has a different number of blocks and potentially a different number of channels
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Adaptive average pooling and a fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # Helper function to create layers with the correct number of blocks and channels
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    # Forward pass through the network
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Instantiating a ResNet50 model


def resnet50():
    # [3, 4, 6, 3] corresponds to the number of blocks in each of the 4 layers for ResNet50
    return ResNet(Bottleneck, [3, 4, 6, 3])


# Example usage
model = resnet50()
print(model)

"""CIFAR candidate, not a claim of architectural novelty or measured accuracy."""

import torch
from torch import nn


class Add(nn.Module):
    def forward(self, left, right):
        return left + right


class Multiply(nn.Module):
    def forward(self, left, right):
        return left * right


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, multiscale=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.multiscale = multiscale
        if multiscale:
            self.context = nn.Sequential(
                nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            hidden = max(8, channels // 16)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.ReLU(),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )
            self.mix = Multiply()
            self.context_add = Add()
            self.context_relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == channels
            else nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        )
        self.add = Add()
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        h = self.relu1(self.bn1(self.conv1(x)))
        if self.multiscale:
            h = self.context_relu(self.context_add(h, self.mix(self.context(h), self.gate(h))))
        return self.relu2(self.add(self.bn2(self.conv2(h)), self.shortcut(x)))


class PDPBlock(nn.Module):
    """Expand/project channel mixing, depthwise context, SE, residual shortcut.

    Established components, not a claim of a new research architecture.
    """

    def __init__(self, in_channels, channels):
        super().__init__()
        expanded = channels * 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded), nn.ReLU(),
            nn.Conv2d(expanded, expanded, 5, padding=2, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded), nn.ReLU(),
            nn.Conv2d(expanded, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        hidden = max(8, channels // 16)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.shortcut = nn.Identity() if in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.multiply, self.add, self.relu = Multiply(), Add(), nn.ReLU()

    def forward(self, x):
        h = self.features(x)
        return self.relu(self.add(self.multiply(h, self.gate(h)), self.shortcut(x)))


class LightNet(nn.Module):
    """Same backbone for 10/100 classes; baseline removes late context branches."""

    def __init__(self, num_classes=10, variant="multiscale", widths=None, depths=None):
        super().__init__()
        widths = widths if widths is not None else ((48, 96, 192, 384) if variant == "hybrid" else (48, 96, 192, 288))
        depths = depths if depths is not None else ((2, 3, 3, 4) if variant == "hybrid" else (2, 3, 3, 2))
        if num_classes not in (10, 100):
            raise ValueError("num_classes must be 10 or 100")
        if variant not in ("baseline", "multiscale", "hybrid"):
            raise ValueError("variant must be baseline, multiscale or hybrid")
        if len(widths) != 4 or len(depths) != 4 or min(*widths, *depths) <= 0:
            raise ValueError("widths/depths must each contain four positive integers")
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(),
        )
        stages = []
        incoming = widths[0]
        for stage, (channels, count) in enumerate(zip(widths, depths)):
            blocks = []
            for block in range(count):
                stride = 2 if stage > 0 and block == 0 else 1
                if variant == "hybrid" and stage == 3:
                    # No third downsampling: retain 8x8 spatial evidence.
                    blocks.append(PDPBlock(incoming, channels))
                else:
                    blocks.append(ResidualBlock(
                        incoming, channels, stride,
                        multiscale=variant == "multiscale" and stage >= 2,
                    ))
                incoming = channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[-1], num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.stages(self.stem(x))), 1))
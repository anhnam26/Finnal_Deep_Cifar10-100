"""Budget-oriented CIFAR CNN, independently implemented from established ideas.

Hierarchical channel splits are inspired by Res2Net (Gao et al., 2019);
pooled channel gates by Squeeze-and-Excitation (Hu et al., 2017). This is
a custom composition and experimental hypothesis, not a research novelty claim.
"""

import torch
from torch import nn

from .operations import Add, Multiply


class ChannelGate(nn.Module):
    """Image-dependent channel recalibration, initialized near identity."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.relu = nn.ReLU()
        self.expand = nn.Conv2d(hidden, channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.scale = Multiply()
        self.product = Multiply()

    def forward(self, x):
        weights = self.sigmoid(self.expand(self.relu(self.reduce(self.pool(x)))))
        return self.product(x, self.scale(weights, 2.))


class ScaleUnit(nn.Module):
    """Dense local extraction followed by a four-split hierarchical refinement.

All four channel chunks are spatially processed. Each chunk after the first
adds the preceding refined chunk before its 3x3 convolution. Concatenation
then 1x1 mixing retains a dense cross-channel path without a second full 3x3.
Strided transitions first establish the new resolution, so split additions
always have identical shapes. An unchanged shortcut is the raw input.
"""

    def __init__(self, incoming, channels, stride=1, gated=False):
        super().__init__()
        self.pre = nn.Sequential(nn.BatchNorm2d(incoming), nn.ReLU())
        self.conv1 = nn.Conv2d(incoming, channels, 3, stride, 1, bias=False)
        self.mid = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU())
        self.spatial = nn.ModuleList([
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1, bias=False)
            for _ in range(4)
        ])
        self.links = nn.ModuleList([Add() for _ in range(3)])
        self.post = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU())
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = ChannelGate(channels) if gated else nn.Identity()
        self.projection = (nn.Identity() if incoming == channels and stride == 1
                           else nn.Conv2d(incoming, channels, 1, stride, bias=False))
        self.add = Add()

    def forward(self, x):
        activated = self.pre(x)
        shortcut = x if isinstance(self.projection, nn.Identity) else self.projection(activated)
        chunks = self.mid(self.conv1(activated)).chunk(4, dim=1)
        refined = []
        for index, (chunk, convolution) in enumerate(zip(chunks, self.spatial)):
            if index:
                chunk = self.links[index - 1](chunk, refined[-1])
            refined.append(convolution(chunk))
        # Chunk/cat/flatten are memory operations, explicitly excluded by profiler.
        residual = self.mix(self.post(torch.cat(refined, dim=1)))
        return self.add(shortcut, self.gate(residual))


class ScaleGateNet(nn.Module):
    """Asymmetric 32/16/8 backbone; gate on/off is the controlled ablation.

Existing interface labels are retained: fusion = local gates in stages2/3;
no_fusion = identical multiscale backbone without gates. There is NO
stage-descriptor fusion and old StageFusionNet weights are incompatible.
"""

    def __init__(self, num_classes=10, variant="fusion", widths=(40, 120, 320), depths=(2, 3, 4)):
        super().__init__()
        if num_classes not in (10, 100):
            raise ValueError("num_classes must be 10 or 100")
        if variant not in ("fusion", "no_fusion"):
            raise ValueError("variant must be fusion or no_fusion")
        if (len(widths) != 3 or len(depths) != 3
                or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                       for v in (*widths, *depths))
                or any(v % 4 for v in widths)):
            raise ValueError("widths/depths require 3 positive integers; widths must be divisible by4")
        self.variant = variant
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        incoming, stages = widths[0], []
        for index, (channels, depth) in enumerate(zip(widths, depths)):
            blocks = []
            for block in range(depth):
                blocks.append(ScaleUnit(incoming, channels, 2 if index and block == 0 else 1,
                                        gated=variant == "fusion" and index > 0))
                incoming = channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.final = nn.Sequential(nn.BatchNorm2d(widths[-1]), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(widths[-1], num_classes)
        self.apply(self._initialize)
        for module in self.modules():
            if isinstance(module, ChannelGate):
                nn.init.normal_(module.expand.weight, std=.01)
                nn.init.zeros_(module.expand.bias)

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
            nn.init.normal_(module.weight, std=.01)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.classifier(torch.flatten(self.final(x), 1))
"""Exam-budget CIFAR candidate composed from established residual/GAP/gating ideas.

Stage-descriptor residual fusion is a design hypothesis, not a novelty or
accuracy claim. The instructor must confirm architecture non-duplication.
"""

import torch
from torch import nn


class Add(nn.Module):
    def forward(self, left, right):
        return left + right


class Multiply(nn.Module):
    def forward(self, left, right):
        return left * right


class ResidualUnit(nn.Module):
    """Preactivated residual unit with a raw identity for unchanged shapes."""

    def __init__(self, incoming, channels, stride=1):
        super().__init__()
        self.pre = nn.Sequential(nn.BatchNorm2d(incoming), nn.ReLU())
        self.conv1 = nn.Conv2d(incoming, channels, 3, stride, 1, bias=False)
        self.mid = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU())
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.projection = (nn.Identity() if incoming == channels and stride == 1
                           else nn.Conv2d(incoming, channels, 1, stride, bias=False))
        self.add = Add()

    def forward(self, x):
        activated = self.pre(x)
        shortcut = x if isinstance(self.projection, nn.Identity) else self.projection(activated)
        return self.add(self.conv2(self.mid(self.conv1(activated))), shortcut)


class Descriptor(nn.Module):
    def __init__(self, incoming, dimension):
        super().__init__()
        self.activation = nn.Sequential(nn.BatchNorm2d(incoming), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = (nn.Identity() if incoming == dimension
                           else nn.Linear(incoming, dimension, bias=False))
        self.normalize = nn.LayerNorm(dimension)

    def forward(self, x):
        pooled = torch.flatten(self.pool(self.activation(x)), 1)
        return self.normalize(self.projection(pooled))


class StageFusionNet(nn.Module):
    """32/16/8 asymmetric residual backbone with optional descriptor fusion.

    z = d3 + sigmoid(g1(d1))*d1 + sigmoid(g2(d2))*d2.
    Each descriptor is normalized after pooling/projection. Early gates start
    near 0.12 so the final-stage path dominates initially. The no_fusion
    ablation removes the early descriptors/gates, not the final head.
    """

    def __init__(self, num_classes=10, variant="fusion", widths=(40, 104, 240), depths=(2, 3, 4)):
        super().__init__()
        if num_classes not in (10, 100):
            raise ValueError("num_classes must be 10 or 100")
        if variant not in ("fusion", "no_fusion"):
            raise ValueError("variant must be fusion or no_fusion")
        if (len(widths) != 3 or len(depths) != 3
                or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                       for v in (*widths, *depths))):
            raise ValueError("widths/depths require 3 positive integers each")
        self.variant = variant
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        incoming, stages = widths[0], []
        for index, (channels, depth) in enumerate(zip(widths, depths)):
            blocks = []
            for block in range(depth):
                blocks.append(ResidualUnit(incoming, channels, 2 if index and block == 0 else 1))
                incoming = channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.final_descriptor = Descriptor(widths[-1], widths[-1])
        if variant == "fusion":
            self.early_descriptors = nn.ModuleList([Descriptor(w, widths[-1]) for w in widths[:-1]])
            self.gates = nn.ModuleList([nn.Sequential(nn.Linear(widths[-1], 1), nn.Sigmoid())
                                        for _ in widths[:-1]])
            self.products = nn.ModuleList([Multiply() for _ in widths[:-1]])
            self.sums = nn.ModuleList([Add() for _ in widths[:-1]])
        self.classifier = nn.Linear(widths[-1], num_classes)
        self.apply(self._initialize)
        if variant == "fusion":
            for gate in self.gates:
                nn.init.constant_(gate[0].bias, -2.)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        intermediate = []
        for stage in self.stages:
            x = stage(x)
            intermediate.append(x)
        descriptor = self.final_descriptor(x)
        if self.variant == "fusion":
            for feature, project, gate, product, add in zip(
                    intermediate[:-1], self.early_descriptors, self.gates, self.products, self.sums):
                early = project(feature)
                descriptor = add(descriptor, product(early, gate(early)))
        return self.classifier(descriptor)
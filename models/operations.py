"""Explicit arithmetic modules so the inference profiler counts custom paths."""

from torch import nn


class Add(nn.Module):
    def forward(self, left, right):
        return left + right


class Multiply(nn.Module):
    def forward(self, left, right):
        return left * right
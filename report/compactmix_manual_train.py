"""COMPACTMIX-L: MỘT FILE ĐỂ TỰ SỬA CONV, FORWARD VÀ TRAIN CIFAR-10/100.

Chạy trong môi trường Microsoft Store Python 3.13 đang dùng:
    python report/compactmix_manual_train.py --check --dataset both

13 class Block01..Block13 ghi TƯỜNG MINH mọi Conv/BN/SE và forward.
Không suy ra số kênh của lớp sau từ lớp trước, không tự vá shape.
Khi sửa một Conv, bạn tự sửa BN, các nhánh, SE, projection, residual,
transition hoặc Linear liên quan. --check chỉ báo lỗi, không sửa giúp.

VÍ DỤ TỰ ĐỔI STAGE 1 TỪ 40 KÊNH THÀNH 48:
  1. CompactMixManual.stem: out_channels=48.
  2. Trong CẢ Block01 và Block02:
     bn_pre=48; expand:48->96; bn_expand=96;
     local/context:48->48, groups=48;
     bn_mix=96; grouped:96->96, groups=8; bn_gate=96;
     SE reduce:96->8; SE expand:8->96; project:96->48.
     Giữ kernel/padding tương ứng 3/1 và 5/2, stride=1.
  3. Transition 1: t1_bn=48, t1_conv.in_channels=48;
     t1_conv.out_channels vẫn 112 nếu Stage 2 chưa đổi.
  4. Chạy --check. Chỉ bắt đầu train khi forward, backward và budget đạt.
  5. Đổi --run-name sau khi sửa code; checkpoint của kiến trúc cũ được giữ.

Đổi kernel context 5->7 phải tự đổi padding 2->3 để giữ H,W.
Đổi projection output phải xử lý cả đường skip trước ADD và block kế tiếp.
Đổi số kênh cuối 256 phải tự sửa head_bn và classifier.in_features.
Không gán accuracy cũ cho model đã chỉnh kiến trúc.

File dùng thư viện chuẩn + torch/torchvision/numpy; không import code dự án.
"""
from __future__ import annotations
import argparse
import copy
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time
import traceback

def ensure_python():
    selected = r"C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe"
    resolved = r"C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
    norm = lambda value: os.path.normcase(os.path.abspath(value))
    if (norm(sys.executable) not in {norm(selected), norm(resolved)}
            or norm(sys.prefix) != norm(sys.base_prefix)
            or sys.version_info[:2] != (3, 13)):
        raise RuntimeError("Chỉ dùng Microsoft Store Python 3.13 đã chọn: " + selected)

ensure_python()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms

class ShapeError(ValueError):
    """Lỗi kết nối cần người học tự sửa thông số."""

def concat_checked(a, b, where):
    if a.shape[0] != b.shape[0] or a.shape[2:] != b.shape[2:]:
        raise ShapeError(f"{where}: local={list(a.shape)}, context={list(b.shape)}. "
                         "Tự sửa kernel/stride/padding để hai nhánh cùng H,W trước cat(dim=1).")
    return torch.cat((a, b), dim=1)

def multiply_checked(feature, gate, where):
    expected = (feature.shape[0], feature.shape[1], 1, 1)
    if tuple(gate.shape) != expected:
        raise ShapeError(f"{where}: feature={list(feature.shape)}, gate={list(gate.shape)}. "
                         f"SE phải trả gate={list(expected)}; sửa Conv cuối của SE.")
    return feature * gate

def residual_checked(x, residual, where):
    if x.shape != residual.shape:
        raise ShapeError(f"{where}: skip={list(x.shape)}, residual={list(residual.shape)}. "
                         "Tự sửa projection/stride/skip để hai tensor bằng shape trước cộng; "
                         "không dùng broadcast để che lỗi.")
    return x + residual

class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("DropPath probability phải thuộc [0,1).")
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1-self.probability
        mask = torch.rand((x.shape[0], 1, 1, 1), device=x.device) < keep
        return x * mask.to(x.dtype) / keep

class ManualBlock(nn.Module):
    # Chỉ dùng làm nhãn cho profiler; không tạo/sửa Conv hoặc số kênh.
    pass



class Block01(ManualBlock):
    """Stage 1; input/output [N,40,32,32]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=40, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=80, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1, dilation=1, groups=40, bias=False)
        self.context = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=5, stride=1, padding=2, dilation=1, groups=40, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=80, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=80, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=80, out_channels=8, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=8, out_channels=80, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=80, out_channels=40, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.0)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block01.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block01.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block01.ADD")


class Block02(ManualBlock):
    """Stage 1; input/output [N,40,32,32]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=40, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=80, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1, dilation=1, groups=40, bias=False)
        self.context = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=5, stride=1, padding=2, dilation=1, groups=40, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=80, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=80, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=80, out_channels=8, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=8, out_channels=80, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=80, out_channels=40, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.008333333333333333)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block02.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block02.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block02.ADD")


class Block03(ManualBlock):
    """Stage 2; input/output [N,112,16,16]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=112, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=224, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1, padding=1, dilation=1, groups=112, bias=False)
        self.context = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=5, stride=1, padding=2, dilation=1, groups=112, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=224, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=224, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=224, out_channels=14, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=14, out_channels=224, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=224, out_channels=112, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.016666666666666666)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block03.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block03.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block03.ADD")


class Block04(ManualBlock):
    """Stage 2; input/output [N,112,16,16]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=112, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=224, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1, padding=1, dilation=1, groups=112, bias=False)
        self.context = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=5, stride=1, padding=2, dilation=1, groups=112, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=224, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=224, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=224, out_channels=14, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=14, out_channels=224, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=224, out_channels=112, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.025000000000000005)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block04.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block04.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block04.ADD")


class Block05(ManualBlock):
    """Stage 2; input/output [N,112,16,16]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=112, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=224, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1, padding=1, dilation=1, groups=112, bias=False)
        self.context = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=5, stride=1, padding=2, dilation=1, groups=112, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=224, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=224, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=224, out_channels=14, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=14, out_channels=224, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=224, out_channels=112, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.03333333333333333)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block05.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block05.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block05.ADD")


class Block06(ManualBlock):
    """Stage 3; input/output [N,256,8,8]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=256, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=False)
        self.context = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, groups=256, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.041666666666666664)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block06.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block06.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block06.ADD")


class Block07(ManualBlock):
    """Stage 3; input/output [N,256,8,8]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=256, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=False)
        self.context = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, groups=256, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.05000000000000001)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block07.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block07.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block07.ADD")


class Block08(ManualBlock):
    """Stage 3; input/output [N,256,8,8]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=256, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=False)
        self.context = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, groups=256, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.05833333333333334)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block08.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block08.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block08.ADD")


class Block09(ManualBlock):
    """Stage 3; input/output [N,256,8,8]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=256, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=False)
        self.context = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, groups=256, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.06666666666666667)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block09.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block09.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block09.ADD")


class Block10(ManualBlock):
    """Stage 3; input/output [N,256,8,8]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=256, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=False)
        self.context = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, groups=256, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.075)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block10.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block10.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block10.ADD")


class Block11(ManualBlock):
    """Stage 3; input/output [N,256,8,8]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=256, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=False)
        self.context = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, groups=256, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.08333333333333333)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block11.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block11.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block11.ADD")


class Block12(ManualBlock):
    """Stage 3; input/output [N,256,8,8]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=256, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=False)
        self.context = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, groups=256, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.09166666666666667)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block12.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block12.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block12.ADD")


class Block13(ManualBlock):
    """Stage 3; input/output [N,256,8,8]. Sửa từng số bên dưới bằng tay."""
    def __init__(self):
        super().__init__()
        self.bn_pre = nn.BatchNorm2d(num_features=256, eps=1e-5, momentum=0.1)
        self.act_pre = nn.SiLU()
        self.expand = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_expand = nn.SiLU()
        self.local = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=False)
        self.context = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, groups=256, bias=False)
        self.bn_mix = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_mix = nn.SiLU()
        self.grouped = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, groups=8, bias=False)
        self.bn_gate = nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)
        self.act_gate = nn.SiLU()
        self.se_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.project = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.drop = DropPath(probability=0.10000000000000002)

    def forward(self, x):
        y = self.bn_pre(x)
        y = self.act_pre(y)
        y = self.expand(y)
        y = self.bn_expand(y)
        y = self.act_expand(y)
        a, b = y.chunk(2, dim=1)
        a = self.local(a)
        b = self.context(b)
        y = concat_checked(a, b, "Block13.concat")
        y = self.bn_mix(y)
        y = self.act_mix(y)
        y = self.grouped(y)
        y = self.bn_gate(y)
        y = self.act_gate(y)
        feature = y
        gate = self.se_pool(feature)
        gate = self.se_reduce(gate)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_sigmoid(gate)
        self._gate_elements = feature.numel()  # Chỉ để đếm FLOPs.
        y = multiply_checked(feature, gate, "Block13.SE")
        y = self.project(y)
        y = self.drop(y)
        return residual_checked(x, y, "Block13.ADD")


class CompactMixManual(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.block01 = Block01()
        self.block02 = Block02()
        self.t1_bn = nn.BatchNorm2d(40, eps=1e-5, momentum=0.1)
        self.t1_act = nn.SiLU()
        self.t1_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.t1_conv = nn.Conv2d(40, 112, kernel_size=1, stride=1, padding=0, bias=False)
        self.block03 = Block03()
        self.block04 = Block04()
        self.block05 = Block05()
        self.t2_bn = nn.BatchNorm2d(112, eps=1e-5, momentum=0.1)
        self.t2_act = nn.SiLU()
        self.t2_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.t2_conv = nn.Conv2d(112, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.block06 = Block06()
        self.block07 = Block07()
        self.block08 = Block08()
        self.block09 = Block09()
        self.block10 = Block10()
        self.block11 = Block11()
        self.block12 = Block12()
        self.block13 = Block13()
        self.head_bn = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.head_act = nn.SiLU()
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(p=0.0)
        # Chỉ số lớp theo dataset; in_features=256 phải tự sửa nếu đổi Stage 3.
        self.classifier = nn.Linear(in_features=256, out_features=num_classes, bias=True)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.block01(x)
        x = self.block02(x)
        x = self.t1_bn(x)
        x = self.t1_act(x)
        x = self.t1_pool(x)
        x = self.t1_conv(x)
        x = self.block03(x)
        x = self.block04(x)
        x = self.block05(x)
        x = self.t2_bn(x)
        x = self.t2_act(x)
        x = self.t2_pool(x)
        x = self.t2_conv(x)
        x = self.block06(x)
        x = self.block07(x)
        x = self.block08(x)
        x = self.block09(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.head_bn(x)
        x = self.head_act(x)
        x = self.head_pool(x)
        x = x.flatten(1)
        x = self.head_dropout(x)
        return self.classifier(x)  # logits, không thêm softmax.

def make_model(num_classes):
    return CompactMixManual(num_classes)

@contextmanager
def input_checks(model):
    handles = []
    def check(name):
        def pre(module, args):
            x = args[0]
            expected = None
            actual = None
            if isinstance(module, nn.Conv2d):
                expected = module.in_channels
                actual = x.shape[1] if x.ndim == 4 else None
            elif isinstance(module, nn.BatchNorm2d):
                expected = module.num_features
                actual = x.shape[1] if x.ndim == 4 else None
            elif isinstance(module, nn.Linear):
                expected, actual = module.in_features, x.shape[-1]
            if expected is not None and actual != expected:
                raise ShapeError(f"{name}: {type(module).__name__} cần {expected} kênh/features "
                                 f"nhưng nhận {actual}, input={list(x.shape)}. "
                                 "Tự sửa output lớp trước hoặc input/BN/Linear của lớp này.")
        return pre
    try:
        for name, module in model.named_modules():
            if not list(module.children()):
                handles.append(module.register_forward_pre_hook(check(name)))
        yield
    finally:
        for handle in handles:
            handle.remove()

def inspect_model(model, num_classes, backward=True):
    """Thực chạy tensor; báo lỗi, không reshape/đệm/sửa kênh tự động."""
    stats = {"parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "flops": 0, "macs": 0, "layers": []}
    handles = []
    mode = model.training
    def capture(name):
        def hook(module, inputs, output):
            n, operations = output.numel(), 0
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                cost = module.in_features if isinstance(module, nn.Linear) else module.in_channels//module.groups*math.prod(module.kernel_size)
                stats["macs"] += n*cost
                operations = 2*n*cost + (n if module.bias is not None else 0)
            elif isinstance(module, nn.BatchNorm2d):
                operations = n*4
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                operations = n*8
            elif isinstance(module, (nn.SiLU, nn.ReLU, nn.GELU, nn.Sigmoid, nn.LeakyReLU)):
                operations = n*(6 if isinstance(module,nn.SiLU) else 8 if isinstance(module,nn.GELU)
                                else 4 if isinstance(module,nn.Sigmoid) else 2 if isinstance(module,nn.LeakyReLU) else 1)
            elif isinstance(module, (nn.AvgPool2d,nn.MaxPool2d,nn.AdaptiveAvgPool2d,nn.AdaptiveMaxPool2d)):
                operations = inputs[0].numel()+n
            elif isinstance(module, ManualBlock):
                operations = n+module._gate_elements  # ADD skip + MUL SE.
            elif not list(module.children()) and not isinstance(module,(DropPath,nn.Dropout,nn.Identity,nn.Flatten)):
                raise ShapeError(f"Profiler chưa hỗ trợ {name}: {type(module).__name__}; cần bổ sung cách đếm FLOPs.")
            stats["flops"] += operations
            if not list(module.children()):
                stats["layers"].append({"name":name,"type":type(module).__name__,
                    "input":list(inputs[0].shape),"output":list(output.shape),
                    "parameters":sum(p.numel() for p in module.parameters())})
        return hook
    try:
        model.cpu().eval()
        for name,module in model.named_modules():
            handles.append(module.register_forward_hook(capture(name)))
        with input_checks(model), torch.no_grad():
            output = model(torch.zeros(1,3,32,32))
        if list(output.shape) != [1,num_classes] or not torch.isfinite(output).all():
            raise ShapeError(f"Output phải là [1,{num_classes}] logits hữu hạn, nhận {list(output.shape)}.")
    finally:
        for handle in handles:
            handle.remove()
        model.train(mode)
    if stats["parameters"] > 6_000_000 or stats["flops"] >= 1_000_000_000:
        raise ShapeError(f"Vượt budget đề: P={stats['parameters']:,}, FLOPs={stats['flops']:,}; "
                         "cần P<=6M và FLOPs<1G.")
    if backward:
        model.train()
        with input_checks(model):
            logits = model(torch.randn(2,3,32,32))
            if logits.shape != (2,num_classes):
                raise ShapeError("Output train/backward không khớp số nhãn.")
            F.cross_entropy(logits,torch.tensor([0,num_classes-1])).backward()
        if not any(p.grad is not None for p in model.parameters()):
            raise ShapeError("Không có gradient; forward đã cắt graph.")
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ShapeError("Gradient không hữu hạn.")
        model.zero_grad(set_to_none=True)
        model.train(mode)
    return stats

def check_command(dataset):
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    for name in (("cifar10","cifar100") if dataset=="both" else (dataset,)):
        classes = 10 if name=="cifar10" else 100
        result = inspect_model(make_model(classes),classes)
        print(f"{name}: FORWARD + BACKWARD OK | P={result['parameters']:,} "
              f"FLOPs={result['flops']:,} | 2 FLOPs/MAC")
        for row in result["layers"]:
            print(f"{row['name']:<28} {row['type']:<18} {row['input']} -> {row['output']} P={row['parameters']:,}")

def main(argv=None):
    parser=argparse.ArgumentParser(description="CompactMix-L explicit Conv/forward in one Python file")
    parser.add_argument("--check",action="store_true",help="Chỉ kiểm tra forward/backward/shape/budget, không train")
    parser.add_argument("--dataset",choices=["both","cifar10","cifar100"],default="both")
    args=parser.parse_args(argv)
    check_command(args.dataset)
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

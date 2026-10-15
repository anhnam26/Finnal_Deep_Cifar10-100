"""COMPACTMIX-L: MỘT FILE ĐỂ TỰ SỬA CONV, FORWARD VÀ TRAIN CIFAR-10/100.

Chạy trong môi trường Microsoft Store Python 3.13 đang dùng:
    python report/compactmix_manual_train.py --check --dataset both
    python -u report/compactmix_manual_train.py --run-name bai_lam
    python -u report/compactmix_manual_train.py --smoke --epochs 2 --dataset both

Train mặc định: CIFAR-10 rồi CIFAR-100, 600 epoch selection 45k/5k.
Sau đó freeze best epoch và loại weights raw/EMA, retrain mới trên 50k rồi test một lần.
--selection-only chỉ train/validation để trình diễn sửa kiến trúc.
Chỉ log/lưu best-last sau một epoch hoàn chỉnh; Ctrl+C lần đầu hoàn tất
epoch rồi dừng. Lặp đúng lệnh để resume; Ctrl+C lần hai dừng ngay.
Checkpoint riêng ở checkpoints/manual_onefile; không tự nạp weights cũ
có tên layer của phiên bản src/models.py.

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


# ======================================================================
# TRAIN SETTINGS: sửa trực tiếp ở đây hoặc dùng CLI --epochs/--lr/...
# Các Conv/BN vẫn phải sửa trực tiếp trong từng Block01..Block13 bên trên.
# ======================================================================
import logging
import re
import subprocess

SCRIPT_PATH = Path(__file__).resolve()
ROOT = next((parent for parent in (SCRIPT_PATH.parent,*SCRIPT_PATH.parents)
             if (parent/"datasets").is_dir()),SCRIPT_PATH.parent)
CHECKPOINT_VERSION = 1
REQUEST_STOP = False
TRAIN_SETTINGS = {
    "epochs":600, "batch_size":128, "workers":2, "seed":42, "split_seed":2026,
    "optimizer":"sgd", "lr":0.1, "weight_decay":0.0005, "momentum":0.9,
    "warmup":5, "scheduler":"cosine", "label_smoothing":0.05,
    "mixup":0.2, "cutmix":1.0, "mix_probability":0.75, "augmentation":"autoaugment",
    "random_erasing":0.0, "ema_decay":0.999, "grad_clip":5.0, "amp":True,
    "deterministic":True, "tensorboard":False,
}
STATS = {
    "cifar10":((.4914,.4822,.4465),(.2470,.2435,.2616)),
    "cifar100":((.5071,.4867,.4408),(.2675,.2565,.2761)),
}

def source_fingerprint():
    # Chỉ đọc chính file đang chạy; không phụ thuộc src/ hay file khác.
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n",b"\n")).hexdigest()

def environment(config):
    return {"python":platform.python_version(),"pytorch":torch.__version__,
            "torchvision":torchvision.__version__,"numpy":np.__version__,
            "cuda":torch.version.cuda,"device":config["device"],"platform":platform.system(),
            "gpu":torch.cuda.get_device_name(0) if config["device"]=="cuda" else None}

def stratified_split(targets, classes, seed):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train, val = [], []
    for label in range(classes):
        indices = np.flatnonzero(targets == label)
        if len(indices) < 2:
            raise ValueError('Each class needs at least two samples')
        rng.shuffle(indices)
        n = max(1, len(indices) // 10)
        val.extend(indices[:n].tolist())
        train.extend(indices[n:].tolist())
    return sorted(train), sorted(val)

def prepare_data(config):
    classes = 10 if config['dataset'] == 'cifar10' else 100
    if config['smoke']:
        source = datasets.FakeData(size=48, image_size=(3, 32, 32), num_classes=classes)
        training, validation = list(range(32)), list(range(32, 48))
    else:
        constructor = datasets.CIFAR10 if classes == 10 else datasets.CIFAR100
        source = constructor(root=config['data_root'], train=True, download=config['download'])
        training, validation = stratified_split(source.targets, classes, config['split_seed'])
    if config['phase'] == 'final':
        training = sorted(training + validation)
        validation = []
    if config['limit_train']:
        training = training[:config['limit_train']]
    if config['limit_val']:
        validation = validation[:config['limit_val']]
    payload = {'training': training, 'validation': validation}
    return source, {**payload, 'classes': classes,
                    'sha256': hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
                    'synthetic': config['smoke'],
                    'subset': bool(config['limit_train'] or config['limit_val'])}

@contextmanager
def sample_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

def image_transforms(config, training):
    mean, std = STATS[config['dataset']]
    ops = []
    if training:
        ops = [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
               transforms.RandomHorizontalFlip()]
        kind = config['augmentation']
        if kind == 'autoaugment':
            ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        elif kind == 'randaugment':
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
        elif kind == 'trivialaugment':
            ops.append(transforms.TrivialAugmentWide())
        elif kind != 'basic':
            raise ValueError(f'Unknown augmentation: {kind}')
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if training and config['random_erasing']:
        ops.append(transforms.RandomErasing(p=config['random_erasing']))
    return transforms.Compose(ops)

class SeededImages(Dataset):
    def __init__(self, source, indices, transform, seed, epoch):
        self.source, self.indices, self.transform = source, indices, transform
        self.seed, self.epoch = seed, epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        seed = (self.seed * 1_000_003 + self.epoch * 97_409 + index) % (2**63-1)
        with sample_rng(seed):
            image, label = self.source[index]
            return self.transform(image), label

def make_loader(source, indices, config, epoch, training=False):
    dataset = SeededImages(source, indices, image_transforms(config, training), config['seed'], epoch)
    generator = torch.Generator().manual_seed(config['seed'] + epoch * 1_000_003)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=training,
                      num_workers=config['workers'], generator=generator,
                      pin_memory=config['device'] == 'cuda', persistent_workers=False,
                      **({'prefetch_factor': 2} if config['workers'] else {}))

def make_optimizer(model, config):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    groups = [{'params': decay, 'weight_decay': config['weight_decay']},
              {'params': no_decay, 'weight_decay': 0.}]
    kind = config['optimizer']
    if kind == 'sgd':
        return torch.optim.SGD(groups, lr=config['lr'], momentum=config['momentum'],
                               nesterov=config['momentum'] > 0)
    if kind == 'adam':
        return torch.optim.Adam(groups, lr=config['lr'], betas=(.9, .999), eps=1e-8)
    if kind == 'adamw':
        return torch.optim.AdamW(groups, lr=config['lr'], betas=(.9, .999), eps=1e-8)
    raise ValueError(f'Unknown optimizer: {kind}')

def learning_rate(config, epoch, batch, batches):
    progress = epoch + batch / batches
    warmup = min(config['warmup'], config['epochs'] / 10)
    if warmup and progress < warmup:
        return config['lr'] * (.1 + .9 * progress / warmup)
    fraction = min(1., (progress-warmup) / max(1., config['epochs']-warmup))
    if config['scheduler'] == 'cosine':
        factor = .001 + .999 * (1 + math.cos(math.pi * fraction)) / 2
    elif config['scheduler'] == 'onecycle':
        factor = (1 - math.cos(math.pi * fraction / .3)) / 2 if fraction < .3 else .001 + .999 * (1 + math.cos(math.pi * (fraction-.3)/.7)) / 2
    else:
        raise ValueError('scheduler must be cosine or onecycle')
    return config['lr'] * max(.001, factor)

def mix_batch(images, labels, classes, config, epoch):
    targets = F.one_hot(labels, classes).float()
    coefficient, paired = 1., labels
    tail = min(30, config['epochs'] // 10)
    if (epoch < config['epochs'] - tail and len(labels) > 1
            and random.random() < config['mix_probability']
            and (config['mixup'] > 0 or config['cutmix'] > 0)):
        permutation = torch.randperm(len(labels), device=images.device)
        paired = labels[permutation]
        cutmix = config['cutmix'] > 0 and (config['mixup'] <= 0 or random.random() < .5)
        if cutmix:
            coefficient = float(np.random.beta(config['cutmix'], config['cutmix']))
            height, width = images.shape[-2:]
            ratio = math.sqrt(1-coefficient)
            cut_w, cut_h = int(width*ratio), int(height*ratio)
            cx, cy = random.randrange(width), random.randrange(height)
            left, right = max(0, cx-cut_w//2), min(width, cx+cut_w//2)
            top, bottom = max(0, cy-cut_h//2), min(height, cy+cut_h//2)
            images = images.clone()
            images[:, :, top:bottom, left:right] = images[permutation, :, top:bottom, left:right]
            coefficient = 1 - (right-left)*(bottom-top)/(height*width)
        else:
            coefficient = float(np.random.beta(config['mixup'], config['mixup']))
            images = coefficient*images + (1-coefficient)*images[permutation]
        targets = coefficient*targets + (1-coefficient)*targets[permutation]
    smoothing = config['label_smoothing']
    return images, targets*(1-smoothing)+smoothing/classes, paired, coefficient

@torch.no_grad()
def update_ema(ema, model, steps, target_decay):
    decay = min(target_decay, (1+steps)/(10+steps))
    current = model.state_dict()
    for name, value in ema.state_dict().items():
        if value.is_floating_point():
            value.lerp_(current[name], 1-decay)
        else:
            value.copy_(current[name])

def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.set_num_threads(min(4, os.cpu_count() or 1))

def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(states):
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'].cpu())
    if states['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in states['cuda']])

@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.train.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another trainer owns {directory}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

def atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb' if binary else 'w',
                            **({} if binary else {'encoding': 'utf-8', 'newline': ''})) as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def atomic_json(path, payload):
    atomic_write(path, lambda stream: json.dump(payload, stream, indent=2, allow_nan=False))

def atomic_torch(path, payload):
    atomic_write(path, lambda stream: torch.save(payload, stream), binary=True)

def load_checkpoint(path):
    # Only deserialize checkpoints written by this trusted repository.
    return torch.load(path, map_location='cpu', weights_only=False)

def cpu_weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

def make_logger(directory):
    logger = logging.getLogger('cifar.' + str(directory))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(directory / 'training.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def close_logger(logger):
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

def duration(seconds):
    seconds = int(seconds)
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'

def train_epoch(model, ema, optimizer, scaler, source, split, config, epoch, global_step):
    loader = make_loader(source, split['training'], config, epoch, training=True)
    model.train()
    loss_sum = correct = seen = skips = 0
    for batch, (images, labels) in enumerate(loader):
        images, labels = images.to(config['device'], non_blocking=True), labels.to(config['device'], non_blocking=True)
        images, targets, paired, coefficient = mix_batch(images, labels, split['classes'], config, epoch)
        lr = learning_rate(config, epoch, batch, len(loader))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(config['device'], enabled=scaler.is_enabled()):
            logits = model(images)
            loss = -(targets * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss; last completed checkpoint preserved')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < previous_scale
        skips += int(skipped)
        if not skipped:
            global_step += 1
            update_ema(ema, model, global_step, config['ema_decay'])
        n = labels.numel()
        loss_sum += loss.item() * n
        prediction = logits.argmax(1)
        correct += (coefficient * prediction.eq(labels).sum().item()
                    + (1-coefficient) * prediction.eq(paired).sum().item())
        seen += n
    return {'train_loss': loss_sum/seen, 'train_mixed_accuracy': 100*correct/seen,
            'lr': lr, 'amp_skips': skips}, global_step

@torch.no_grad()
def evaluate(model, loader, device, extended=False):
    model.eval()
    loss_sum = correct = seen = 0
    confusion = None
    top5 = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
        predictions = logits.argmax(1)
        correct += predictions.eq(labels).sum().item()
        seen += labels.numel()
        if extended:
            classes = logits.shape[1]
            if confusion is None:
                confusion = torch.zeros(classes, classes, dtype=torch.long)
            confusion += torch.bincount((labels*classes+predictions).cpu(), minlength=classes**2).reshape(classes, classes)
            top5 += logits.topk(min(5, classes), 1).indices.eq(labels[:, None]).any(1).sum().item()
    if seen == 0:
        raise ValueError('Evaluation split is empty')
    result = {'loss': loss_sum/seen, 'accuracy': 100*correct/seen}
    if extended:
        tp = confusion.diag().float()
        recall = tp / confusion.sum(1).clamp_min(1)
        precision = tp / confusion.sum(0).clamp_min(1)
        f1 = 2*precision*recall / (precision+recall).clamp_min(1e-12)
        result.update(top5_accuracy=100*top5/seen, macro_f1=f1.mean().item(),
                      per_class_recall=recall.tolist(), confusion_matrix=confusion.tolist())
    return result

def write_history(directory, history):
    if not history:
        return
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    atomic_write(directory / 'metrics.csv', write)

def validate_config(config):
    if config['phase'] not in ('selection', 'final'):
        raise ValueError('phase must be selection or final')
    if min(config['epochs'], config['batch_size']) < 1 or config['workers'] < 0:
        raise ValueError('Invalid epoch/batch/worker settings')
    if not 0 <= config['label_smoothing'] < 1 or not 0 <= config['mix_probability'] <= 1:
        raise ValueError('Invalid label smoothing or mixing probability')
    if config['lr'] <= 0 or config['weight_decay'] < 0 or config['grad_clip'] <= 0:
        raise ValueError('Invalid learning rate, weight decay or gradient clipping')
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in the selected Python; use --device cpu for debugging')

def run_experiment(config, stop_after_epochs=None, epoch_callback=None):
    config = copy.deepcopy(config)
    validate_config(config)
    directory = Path(config['run_dir']).resolve()
    config['run_dir'] = str(directory)
    with run_lock(directory):
        logger = make_logger(directory)
        writer = None
        try:
            identity, env = source_fingerprint(), environment(config)
            last = directory / 'last.pt'
            if not last.exists() and any((directory / name).exists()
                                         for name in ('best.pt', 'metrics.csv', 'result.json')):
                raise ValueError('Run artifacts exist but last.pt is missing; existing files preserved. '
                                 'Restore last.pt or use a new --run-name')
            previous = load_checkpoint(last) if last.exists() else None
            if previous:
                if previous['version'] != CHECKPOINT_VERSION or previous['source_sha256'] != identity:
                    raise ValueError('Changed training source/version; use a new --run-name, existing checkpoints preserved')
                if previous['config'] != config:
                    raise ValueError('Changed configuration; resume rejected. Restore settings or use --run-name')
                if previous['environment'] != env:
                    raise ValueError('Changed runtime environment; exact resume requires the recorded environment')
                if (directory / 'result.json').exists() and previous['epoch'] == config['epochs']:
                    return json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            seed_everything(config['seed'], config['deterministic'])
            source, split = prepare_data(config)
            if previous and previous['split_sha256'] != split['sha256']:
                raise ValueError('Changed dataset split; resume rejected')
            model = make_model(split['classes'])
            profile = inspect_model(model, split['classes'], backward=False)
            model = model.to(config['device'])
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            optimizer = make_optimizer(model, config)
            scaler = torch.amp.GradScaler('cuda', enabled=config['device'] == 'cuda' and config['amp'], init_scale=4096.)
            history, best_weights = [], None
            best_needs_repair = False
            epoch, global_step, best_accuracy, best_epoch, best_kind, elapsed_seconds = 0, 0, -1., 0, 'raw', 0.
            if previous:
                model.load_state_dict(previous['model'])
                ema.load_state_dict(previous['ema'])
                optimizer.load_state_dict(previous['optimizer'])
                scaler.load_state_dict(previous['scaler'])
                epoch, global_step = previous['epoch'], previous['global_step']
                history = previous['history']
                best_accuracy, best_epoch, best_kind = previous['best_accuracy'], previous['best_epoch'], previous['best_kind']
                best_weights, elapsed_seconds = previous['best_weights'], previous['elapsed_seconds']
                restore_rng(previous['rng'])
                best_path = directory / 'best.pt'
                try:
                    best_document = load_checkpoint(best_path) if best_path.exists() else None
                except Exception:
                    best_document = None
                if (best_document is None or best_document['epoch'] != best_epoch
                        or best_document['accuracy'] != best_accuracy):
                    best_needs_repair = True
            try:
                git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                git_commit = 'unavailable'
            atomic_json(directory / 'config.json', config)
            atomic_json(directory / 'split.json', split)
            atomic_json(directory / 'metadata.json', {'environment': env, 'python_executable': sys.executable,
                        'git_commit': git_commit, 'source_sha256': identity, 'profile': profile,
                        'started_at_utc': previous['started_at_utc'] if previous else datetime.now(timezone.utc).isoformat()})
            metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
            completed_this_call = 0
            while epoch < config['epochs']:
                started = time.monotonic()
                if config['device'] == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                metrics, global_step = train_epoch(model, ema, optimizer, scaler, source, split, config, epoch, global_step)
                if config['phase'] == 'selection':
                    loader = make_loader(source, split['validation'], config, 0)
                    raw = evaluate(model, loader, config['device'])
                    averaged = evaluate(ema, loader, config['device'])
                    score, kind, selected = raw['accuracy'], 'raw', model
                    if averaged['accuracy'] > score:
                        score, kind, selected = averaged['accuracy'], 'ema', ema
                    metrics.update(val_loss=raw['loss'], val_accuracy=raw['accuracy'],
                                   ema_val_loss=averaged['loss'], ema_val_accuracy=averaged['accuracy'])
                    improved = score > best_accuracy
                else:
                    selected = ema if config['final_weights'] == 'ema' else model
                    score, kind, improved = -1., config['final_weights'] + '_fixed_epoch', True
                epoch += 1
                if improved:
                    best_accuracy, best_epoch, best_kind, best_weights = score, epoch, kind, cpu_weights(selected)
                record = {'epoch': epoch, **metrics, 'best_val_accuracy': best_accuracy, 'best_epoch': best_epoch}
                history.append(record)
                seconds = time.monotonic() - started
                elapsed_seconds += seconds
                if config['tensorboard'] and writer is None:
                    from torch.utils.tensorboard import SummaryWriter
                    writer = SummaryWriter(str(directory / 'tensorboard'), purge_step=epoch)
                payload = {'version': CHECKPOINT_VERSION, 'source_sha256': identity, 'environment': env,
                           'config': config, 'split_sha256': split['sha256'], 'epoch': epoch,
                           'global_step': global_step, 'model': cpu_weights(model), 'ema': cpu_weights(ema),
                           'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                           'rng': capture_rng(), 'history': history, 'best_accuracy': best_accuracy,
                           'best_epoch': best_epoch, 'best_kind': best_kind, 'best_weights': best_weights,
                           'elapsed_seconds': elapsed_seconds, 'started_at_utc': metadata['started_at_utc']}
                best_saved = improved or best_needs_repair
                if best_saved:
                    atomic_torch(directory / 'best.pt', {'model': best_weights, 'epoch': best_epoch,
                                 'accuracy': best_accuracy, 'kind': best_kind, 'config': config})
                    best_needs_repair = False
                atomic_torch(last, payload)
                write_history(directory, history)
                if writer:
                    for key, value in record.items():
                        writer.add_scalar(key, value, epoch)
                    writer.flush()
                vram = torch.cuda.max_memory_allocated()/2**30 if config['device'] == 'cuda' else 0.
                logger.info('%s %s %s/%s(base_lr=%g) | EPOCH %d/%d | loss=%.4f mixed_acc=%.2f%% val_raw=%s val_ema=%s best=%s @%d | lr=%.6g time=%s total_ETA=%s VRAM=%.2fGB AMP_skips=%d | saved last.pt%s',
                            config['dataset'], config['phase'], config['model']['family'], config['optimizer'], config['lr'],
                            epoch, config['epochs'], metrics['train_loss'], metrics['train_mixed_accuracy'],
                            f"{metrics['val_accuracy']:.2f}%" if 'val_accuracy' in metrics else 'sealed',
                            f"{metrics['ema_val_accuracy']:.2f}%" if 'ema_val_accuracy' in metrics else 'sealed',
                            f'{best_accuracy:.2f}%' if best_accuracy >= 0 else 'fixed final epoch', best_epoch,
                            metrics['lr'], duration(seconds), duration(elapsed_seconds/epoch*(config['epochs']-epoch)),
                            vram, metrics['amp_skips'], ' + best.pt' if best_saved else '')
                completed_this_call += 1
                if epoch_callback:
                    epoch_callback(epoch, best_accuracy)
                if REQUEST_STOP or (stop_after_epochs and completed_this_call >= stop_after_epochs):
                    if epoch < config['epochs']:
                        return {'status': 'interrupted', 'epoch': epoch, 'history': history}
                    break
            result = {'status': 'complete', 'experiment_id': directory.name, 'dataset': config['dataset'],
                      'model_name': config['model']['family'], 'architecture_version': 'compactmix-manual-onefile-v1',
                      'seed': config['seed'], 'parameters': profile['parameters'], 'flops': profile['flops'],
                      'optimizer': config['optimizer'], 'learning_rate': config['lr'], 'weight_decay': config['weight_decay'],
                      'scheduler': config['scheduler'], 'augmentation': config['augmentation'], 'epochs': epoch,
                      'best_val_accuracy': best_accuracy if config['phase'] == 'selection' else None,
                      'best_epoch': best_epoch, 'best_kind': best_kind,
                      'final_val_accuracy': max(history[-1]['val_accuracy'], history[-1]['ema_val_accuracy']) if config['phase'] == 'selection' else None,
                      'test_accuracy': None, 'training_time': elapsed_seconds, 'checkpoint': str(directory / 'best.pt'),
                      'synthetic': split['synthetic'], 'subset': split['subset'], 'phase': config['phase'],
                      'history': history, 'config': config, 'profile': profile, 'metadata': metadata,
                      'notes': 'Synthetic/subset checks are not benchmark evidence' if split['synthetic'] or split['subset'] else ''}
            atomic_json(directory / 'result.json', result)
            return result
        finally:
            if writer:
                writer.close()
            close_logger(logger)


def read_result(directory):
    path = Path(directory)/"result.json"
    if not path.exists():
        raise ValueError("Run chưa hoàn tất, chưa có result.json.")
    return json.loads(path.read_text(encoding="utf-8"))

def evaluate_final(directory, verbose=True):
    directory = Path(directory).resolve()
    with run_lock(directory):
        result = read_result(directory)
        config = result["config"]
        freeze = json.loads((directory/"frozen.json").read_text(encoding="utf-8"))
        if result["status"]!="complete" or result["phase"]!="final":
            raise ValueError("Test yêu cầu completed frozen final run.")
        if result["synthetic"] or result["subset"]:
            raise ValueError("Smoke/subset không được dùng làm benchmark test.")
        if freeze["source_sha256"]!=source_fingerprint() or freeze["config"]!=config:
            raise ValueError("Source/config đã đổi sau freeze; không đánh giá checkpoint khác kiến trúc.")
        if result["metadata"]["environment"] != environment(config):
            raise ValueError("Runtime đã đổi so với final run.")
        checkpoint_path=directory/"best.pt"
        digest=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if result["test_accuracy"] is not None:
            if result["test_checkpoint_sha256"]!=digest:
                raise ValueError("Checkpoint đã đổi sau test; không tái sử dụng điểm số.")
        else:
            classes=10 if config["dataset"]=="cifar10" else 100
            model=make_model(classes).to(config["device"])
            model.load_state_dict(load_checkpoint(checkpoint_path)["model"],strict=True)
            constructor=datasets.CIFAR10 if classes==10 else datasets.CIFAR100
            source=constructor(root=config["data_root"],train=False,download=config["download"])
            loader=make_loader(source,list(range(len(source))),config,0,training=False)
            metrics=evaluate(model,loader,config["device"],extended=True)
            result.update(test_accuracy=metrics["accuracy"],test_metrics=metrics,
                          test_checkpoint_sha256=digest)
            atomic_json(directory/"result.json",result)
        if verbose:
            print(json.dumps({"dataset":config["dataset"],"test_accuracy":result["test_accuracy"],
                              "checkpoint_sha256":digest},ensure_ascii=False))
        return result

def finalize(selection_directory, stop_after_epochs=None):
    selection_directory=Path(selection_directory).resolve()
    result=read_result(selection_directory)
    if result["status"]!="complete" or result["phase"]!="selection":
        raise ValueError("Cần selection hoàn tất để freeze và retrain final.")
    if result["synthetic"] or result["subset"]:
        raise ValueError("Không finalize smoke/subset thành benchmark.")
    if result["metadata"]["source_sha256"] != source_fingerprint():
        raise ValueError("Selection source đã đổi; chạy selection mới với --run-name mới trước khi freeze.")
    if result["metadata"]["environment"] != environment(result["config"]):
        raise ValueError("Selection runtime đã đổi; không freeze recipe của runtime khác.")
    config=copy.deepcopy(result["config"])
    config.update(phase="final",epochs=result["best_epoch"],final_weights=result["best_kind"],
                  run_dir=str(selection_directory.with_name(selection_directory.name+"_final")))
    directory=Path(config["run_dir"])
    freeze={"selection_run":str(selection_directory),"selection_best_epoch":result["best_epoch"],
            "selection_validation_accuracy":result["best_val_accuracy"],"config":config,
            "source_sha256":source_fingerprint()}
    path=directory/"frozen.json"
    directory.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=freeze:
            raise ValueError("Final recipe/source đã đổi; giữ checkpoint cũ, dùng --run-name mới.")
    else:
        atomic_json(path,freeze)
    trained=run_experiment(config,stop_after_epochs)
    if trained["status"]=="interrupted" or REQUEST_STOP:
        return trained
    return evaluate_final(directory,verbose=False)

def parser():
    p=argparse.ArgumentParser(description="Một file: 13 block Conv tường minh, forward, kiểm tra shape, train/resume CIFAR")
    p.add_argument("--check",action="store_true",help="In shape từng lớp, chạy forward/backward CPU rồi thoát")
    p.add_argument("--dataset",choices=["both","cifar10","cifar100"],default="both")
    p.add_argument("--run-name",default="manual")
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--data-root",type=Path,default=ROOT/"datasets")
    p.add_argument("--output-root",type=Path)
    p.add_argument("--optimizer",choices=["sgd","adam","adamw"])
    for name,kind in (("epochs",int),("batch-size",int),("workers",int),
                      ("seed",int),("lr",float),("weight-decay",float)):
        p.add_argument("--"+name,type=kind)
    p.add_argument("--download",action="store_true",help="Cho phép torchvision tải dữ liệu CIFAR còn thiếu")
    p.add_argument("--smoke",action="store_true",help="FakeData 48 ảnh; không phải accuracy benchmark")
    p.add_argument("--stop-after-epochs",type=int,help="Dừng ở ranh giới sau N epoch mới trong một run")
    p.add_argument("--selection-only",action="store_true",help="Chưa retrain 50k/test")
    p.add_argument("--finalize",type=Path,help="Selection run đã xong: freeze rồi retrain 50k/test")
    p.add_argument("--evaluate-final",type=Path,help="Đọc hoặc đo test của frozen final đã hoàn tất")
    return p

def configs_from_args(args):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",args.run_name):
        raise ValueError("run-name chỉ gồm chữ, số, dấu chấm/gạch dưới/gạch ngang.")
    if args.stop_after_epochs is not None and args.stop_after_epochs<1:
        raise ValueError("stop-after-epochs phải >=1.")
    names=("cifar10","cifar100") if args.dataset=="both" else (args.dataset,)
    output_root=args.output_root or (ROOT/"experiments/manual_onefile_smoke"
                                    if args.smoke else ROOT/"checkpoints/manual_onefile")
    configs=[]
    for dataset in names:
        config=copy.deepcopy(TRAIN_SETTINGS)
        config.update(dataset=dataset,model={"family":"compactmix_manual"},
                      phase="selection",final_weights="ema",device=args.device,
                      data_root=str(args.data_root.resolve()),download=args.download,
                      smoke=args.smoke,limit_train=0,limit_val=0)
        if args.smoke:
            config.update(epochs=2,batch_size=4,workers=0)
        for key in ("epochs","batch_size","workers","seed","lr","weight_decay","optimizer"):
            value=getattr(args,key)
            if value is not None:
                config[key]=value
        if args.optimizer in ("adam","adamw"):
            if args.lr is None:
                config["lr"]=.001
            if args.weight_decay is None:
                config["weight_decay"]=.0001
        label=f"{args.run_name}_{dataset}_{config['optimizer']}_lr{config['lr']:g}_seed{config['seed']}"
        config["run_dir"]=str((output_root/label).resolve())
        validate_config(config)
        configs.append(config)
    return configs

def main(argv=None):
    global REQUEST_STOP
    args=parser().parse_args(argv)
    if args.check:
        check_command(args.dataset)
        return 0
    if args.evaluate_final:
        evaluate_final(args.evaluate_final)
        return 0
    def handler(signum,frame):
        global REQUEST_STOP
        if REQUEST_STOP:
            raise KeyboardInterrupt
        REQUEST_STOP=True
    previous={}
    for kind in (signal.SIGINT,signal.SIGTERM):
        previous[kind]=signal.signal(kind,handler)
    try:
        if args.finalize:
            selection=read_result(args.finalize)
            classes=10 if selection["dataset"]=="cifar10" else 100
            inspect_model(make_model(classes),classes,backward=True)
            finalize(args.finalize,args.stop_after_epochs)
            return 0
        configs=configs_from_args(args)
        # Preflight im lặng; không tải ảnh, không tạo checkpoint, không tự sửa model.
        torch.set_num_threads(min(4,os.cpu_count() or 1))
        for config in configs:
            inspect_model(make_model(10 if config["dataset"]=="cifar10" else 100),
                          10 if config["dataset"]=="cifar10" else 100,backward=True)
        complete=[]
        for config in configs:
            result=run_experiment(config,args.stop_after_epochs)
            if result["status"]=="interrupted" or REQUEST_STOP:
                return 0
            complete.append(result)
        if not args.selection_only and not args.smoke:
            for result in complete:
                final=finalize(result["config"]["run_dir"],args.stop_after_epochs)
                if final["status"]=="interrupted" or REQUEST_STOP:
                    break
        return 0
    finally:
        for kind,old in previous.items():
            signal.signal(kind,old)

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Dừng ngay; chạy lại đúng lệnh để tiếp tục từ last.pt của epoch hoàn chỉnh.",file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        # Traceback trỏ đúng dòng Conv/BN/forward cần người học tự sửa.
        traceback.print_exc()
        raise SystemExit(1)

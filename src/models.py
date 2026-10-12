"""CIFAR-native architectures and an explicit inference operation budget."""
import math
import torch
from torch import nn


def norm_layer(channels, kind='bn'):
    if kind == 'bn':
        return nn.BatchNorm2d(channels)
    if kind == 'gn':
        return nn.GroupNorm(math.gcd(8, channels), channels)
    if kind == 'ln':
        return nn.GroupNorm(1, channels)  # Spatial/channel layer normalization.
    raise ValueError(f'Unknown normalization: {kind}')


def activation(kind='silu'):
    return {'silu': nn.SiLU, 'relu': nn.ReLU, 'gelu': nn.GELU}[kind]()


class DropPath(nn.Module):
    def __init__(self, probability=0.):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError('Drop path must lie in [0,1)')
        self.probability = probability

    def forward(self, x):
        if not self.training or not self.probability:
            return x
        keep = 1 - self.probability
        mask = torch.rand((x.shape[0], 1, 1, 1), device=x.device) < keep
        return x * mask.to(x.dtype) / keep


class ChannelGate(nn.Module):
    def __init__(self, channels, kind='se'):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.kind = kind
        if kind == 'se':
            hidden = max(8, channels // 16)
            self.gate = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.SiLU(),
                                      nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        elif kind == 'eca':
            self.gate = nn.Conv1d(1, 1, 3, padding=1, bias=False)
            self.sigmoid = nn.Sigmoid()
        else:
            raise ValueError(f'Unknown gate: {kind}')

    def forward(self, x):
        pooled = self.pool(x)
        if self.kind == 'se':
            gate = self.gate(pooled)
        else:
            gate = self.sigmoid(self.gate(pooled.flatten(2).transpose(1, 2)))
            gate = gate.transpose(1, 2).unsqueeze(-1)
        return x * gate


class MixBlock(nn.Module):
    """Two spatial scales followed by grouped mixing and dense projection."""
    def __init__(self, channels, drop_path=0., context_kernel=5, expansion=2,
                 attention='se', norm='bn', act='silu', residual_scale=1.):
        super().__init__()
        expanded = channels * expansion
        if expanded % 16 or context_kernel not in (3, 5, 7):
            raise ValueError('Expanded channels must be divisible by 16; kernel in 3/5/7')
        self.pre = nn.Sequential(norm_layer(channels, norm), activation(act),
                                 nn.Conv2d(channels, expanded, 1, bias=False),
                                 norm_layer(expanded, norm), activation(act))
        half = expanded // 2
        self.local = nn.Conv2d(half, half, 3, padding=1, groups=half, bias=False)
        self.context = nn.Conv2d(half, half, context_kernel, padding=context_kernel // 2,
                                 groups=half, bias=False)
        self.mix = nn.Sequential(norm_layer(expanded, norm), activation(act),
                                 nn.Conv2d(expanded, expanded, 3, padding=1, groups=8, bias=False),
                                 norm_layer(expanded, norm), activation(act),
                                 ChannelGate(expanded, attention) if attention != 'none' else nn.Identity(),
                                 nn.Conv2d(expanded, channels, 1, bias=False))
        self.drop = DropPath(drop_path)
        self.residual_scale = residual_scale

    def forward(self, x):
        a, b = self.pre(x).chunk(2, dim=1)
        residual = self.mix(torch.cat((self.local(a), self.context(b)), dim=1))
        if self.residual_scale != 1:
            residual = residual * self.residual_scale
        return x + self.drop(residual)


class GlobalPool(nn.Module):
    def __init__(self, kind='avg'):
        super().__init__()
        self.kind = kind
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maximum = nn.AdaptiveMaxPool2d(1) if kind == 'avgmax' else None
        if kind not in ('avg', 'avgmax'):
            raise ValueError('pool must be avg or avgmax')

    def forward(self, x):
        pooled = self.avg(x)
        if self.maximum is not None:
            pooled = (pooled + self.maximum(x)) * .5
        return pooled.flatten(1)


class CompactMixNet(nn.Module):
    def __init__(self, classes, widths=(40, 112, 256), depths=(2, 3, 8),
                 drop_path=.1, dropout=0., stem='simple', pool='avg',
                 norm='bn', activation='silu', **block_options):
        super().__init__()
        if len(widths) != 3 or len(depths) != 3 or min(depths) < 1:
            raise ValueError('CompactMix-L needs three nonempty stages')
        self.widths, self.depths = list(widths), list(depths)
        self.resolutions = [32, 16, 8]
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1, bias=False))
        if stem == 'stacked':
            self.stem.extend([norm_layer(widths[0], norm), globals()['activation'](activation),
                              nn.Conv2d(widths[0], widths[0], 3, padding=1, bias=False)])
        elif stem != 'simple':
            raise ValueError('stem must be simple or stacked')
        stages, position = [], 0
        for stage, (width, depth) in enumerate(zip(widths, depths)):
            layers = []
            if stage:
                layers += [norm_layer(widths[stage-1], norm), globals()['activation'](activation),
                           nn.AvgPool2d(2), nn.Conv2d(widths[stage-1], width, 1, bias=False)]
            for _ in range(depth):
                layers.append(MixBlock(width, drop_path * position / max(1, sum(depths)-1),
                                       norm=norm, act=activation, **block_options))
                position += 1
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(norm_layer(widths[-1], norm), globals()['activation'](activation),
                                  GlobalPool(pool), nn.Dropout(dropout), nn.Linear(widths[-1], classes))
        initialize(self)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


class PreActBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1, dropout=0.):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(incoming)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.shortcut = nn.Conv2d(incoming, outgoing, 1, stride, bias=False) if incoming != outgoing or stride != 1 else None

    def forward(self, x):
        pre = self.relu1(self.bn1(x))
        skip = self.shortcut(pre) if self.shortcut is not None else x
        return skip + self.conv2(self.drop(self.relu2(self.bn2(self.conv1(pre)))))


class ResidualNet(nn.Module):
    def __init__(self, classes, widths=(64, 128, 256), depths=(2, 2, 2), dropout=.0):
        super().__init__()
        self.widths, self.depths, self.resolutions = list(widths), list(depths), [32, 16, 8]
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        blocks, incoming = [], 16
        for stage, (width, depth) in enumerate(zip(widths, depths)):
            for position in range(depth):
                blocks.append(PreActBlock(incoming, width, 2 if stage and position == 0 else 1, dropout))
                incoming = width
        self.stages = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.BatchNorm2d(incoming), nn.ReLU(), GlobalPool(), nn.Linear(incoming, classes))
        initialize(self)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, drop_path):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels*4), nn.GELU(), nn.Linear(channels*4, channels))
        self.scale = nn.Parameter(torch.full((channels,), 1e-6))
        self.drop = DropPath(drop_path)

    def forward(self, x):
        y = self.depthwise(x).permute(0, 2, 3, 1)
        y = self.mlp(self.norm(y)) * self.scale
        return x + self.drop(y.permute(0, 3, 1, 2))


class CIFARConvNeXt(nn.Module):
    def __init__(self, classes, widths=(48, 96, 192), depths=(2, 3, 6), drop_path=.1):
        super().__init__()
        self.widths, self.depths, self.resolutions = list(widths), list(depths), [32, 16, 8]
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1)
        layers, index = [], 0
        for stage, (width, depth) in enumerate(zip(widths, depths)):
            if stage:
                layers += [nn.GroupNorm(1, widths[stage-1]), nn.Conv2d(widths[stage-1], width, 2, 2)]
            for _ in range(depth):
                layers.append(ConvNeXtBlock(width, drop_path * index / max(1, sum(depths)-1)))
                index += 1
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(GlobalPool(), nn.LayerNorm(widths[-1]), nn.Linear(widths[-1], classes))
        initialize(self)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def initialize(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            nn.init.zeros_(module.bias)


def build_model(config, classes):
    config = dict(config)
    family = config.pop('family', 'compactmix')
    if config.pop('pretrained', False):
        raise ValueError('Exam pipeline is from scratch; pretrained experiments must be separate')
    if family == 'compactmix':
        return CompactMixNet(classes, **config)
    if family == 'wrn':
        return ResidualNet(classes, **config)
    if family == 'preactresnet':
        return ResidualNet(classes, **{'widths': (32, 64, 128), 'depths': (3, 3, 3), **config})
    if family == 'convnext':
        return CIFARConvNeXt(classes, **config)
    raise ValueError(f'Unknown family {family}; choose compactmix/wrn/preactresnet/convnext')


def profile_model(model, enforce=True):
    counts = {'flops': 0, 'macs': 0}
    handles = []
    def hook(module, inputs, output):
        n = output.numel()
        operations = 0
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            cost = module.in_features if isinstance(module, nn.Linear) else module.in_channels // module.groups * math.prod(module.kernel_size)
            macs = n * cost
            counts['macs'] += macs
            operations = macs * 2 + (n if module.bias is not None else 0)
        elif isinstance(module, nn.BatchNorm2d):
            operations = n * 4
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            operations = n * 8
        elif isinstance(module, (nn.SiLU, nn.GELU, nn.Sigmoid, nn.ReLU)):
            operations = n * (6 if isinstance(module, nn.SiLU) else 8 if isinstance(module, nn.GELU) else 4 if isinstance(module, nn.Sigmoid) else 1)
        elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
            operations = inputs[0].numel() + n
        elif isinstance(module, (MixBlock, PreActBlock, ChannelGate)):
            operations = n * (2 if isinstance(module, MixBlock) and module.residual_scale != 1 else 1)
        elif isinstance(module, ConvNeXtBlock):
            operations = n * 2  # LayerScale multiply and residual add.
        elif isinstance(module, GlobalPool) and module.kind == 'avgmax':
            operations = n * 2
        counts['flops'] += operations
    mode = model.training
    try:
        model.eval()
        for module in model.modules():
            handles.append(module.register_forward_hook(hook))
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(mode)
    result = {**counts, 'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'input': [1, 3, 32, 32], 'widths': model.widths,
              'depths': model.depths, 'resolutions': model.resolutions,
              'convention': '2 FLOPs/MAC; analytical estimate including norm/activations/pools/adds/gates; memory operations excluded'}
    if enforce and (result['parameters'] > 6_000_000 or result['flops'] >= 1_000_000_000):
        raise ValueError(f'Model exceeds exam budget: {result}')
    return result

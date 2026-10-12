"""Reference families with CIFAR stems, never pretrained in exam mode."""
import torch
from torch import nn

REFERENCE_MODELS = {
    'resnet': 'resnet10t', 'resnext': 'resnext50_32x4d',
    'convnextv2': 'convnextv2_atto', 'efficientnet': 'efficientnet_b0',
    'regnet': 'regnetx_002', 'mobilenet': 'mobilenetv3_small_100',
    'vit': 'vit_tiny_patch16_224', 'deit': 'deit_tiny_patch16_224',
    'swin': 'swin_tiny_patch4_window7_224', 'hybrid': 'mobilevit_xxs'}


def replace_first_conv(module, kernel=None):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            kernel = kernel or child.kernel_size[0]
            replacement = nn.Conv2d(child.in_channels, child.out_channels, kernel,
                                     stride=1, padding=kernel//2, groups=child.groups,
                                     bias=child.bias is not None)
            nn.init.kaiming_normal_(replacement.weight, nonlinearity='relu')
            if replacement.bias is not None:
                nn.init.zeros_(replacement.bias)
            module._modules[name] = replacement
            return True
        if replace_first_conv(child, kernel):
            return True
    return False


class TimmCifarModel(nn.Module):
    def __init__(self, family, classes, **options):
        super().__init__()
        import timm
        name = options.pop('name', REFERENCE_MODELS[family])
        kwargs = {}
        factor = 1
        if family == 'convnextv2':
            kwargs.update(stem_type='overlap', patch_size=1)
        elif family in ('vit', 'deit'):
            kwargs.update(img_size=32, patch_size=4)
        elif family == 'swin':
            kwargs.update(img_size=32, patch_size=2, window_size=4,
                          embed_dim=32, depths=(2,2,2,2), num_heads=(2,4,8,8))
        elif family == 'resnext':
            kwargs.update(cardinality=8, base_width=8, channels=(16,32,64,128),
                          stem_type='deep', stem_width=32)
        kwargs.update(options)
        model = timm.create_model(name, pretrained=False, num_classes=classes, **kwargs)
        if family in ('resnet', 'resnext'):
            if isinstance(model.conv1, nn.Conv2d):
                old = model.conv1
                model.conv1 = nn.Conv2d(3, old.out_channels, 3, padding=1, bias=False)
                nn.init.kaiming_normal_(model.conv1.weight, nonlinearity='relu')
            elif not replace_first_conv(model.conv1, 3):
                raise ValueError('Unsupported residual stem')
            model.maxpool = nn.Identity()
            factor = 4
        elif family in ('efficientnet', 'mobilenet'):
            model.conv_stem.stride = (1, 1)
            factor = 2
        elif family in ('regnet', 'hybrid'):
            if not replace_first_conv(model.stem):
                raise ValueError('Unsupported reference stem')
            factor = 2
        self.backbone = model
        self.is_reference = True
        info = getattr(model, 'feature_info', [])
        if hasattr(info, 'get_dicts'):
            info = info.get_dicts()
        if info:
            self.widths = [entry['num_chs'] for entry in info]
            self.resolutions = [32 // max(1, entry['reduction']//factor) for entry in info]
        else:
            self.widths = [getattr(model, 'num_features', 192)]
            self.resolutions = [8]
        if family in ('resnet', 'resnext'):
            self.depths = [len(getattr(model, 'layer'+str(i))) for i in range(1,5)]
        elif hasattr(model, 'stages'):
            self.depths = [len(stage.blocks) if hasattr(stage, 'blocks') else 1 for stage in model.stages]
        elif hasattr(model, 'blocks'):
            self.depths = [len(stage) if isinstance(stage, nn.Sequential) else 1 for stage in model.blocks]
        elif hasattr(model, 'layers'):
            self.depths = [len(stage.blocks) for stage in model.layers]
        else:
            self.depths = [1] * len(self.widths)

    def forward(self, x):
        return self.backbone(x)

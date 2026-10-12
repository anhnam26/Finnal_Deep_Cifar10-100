"""Optimizer, schedule and explicitly configurable augmentation recipe."""
import math
import random
import numpy as np
import torch
from torch.nn import functional as F


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

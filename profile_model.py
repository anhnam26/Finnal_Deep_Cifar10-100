"""Transparent inference count for every operation used in LightNet.

Conv/linear multiply-add = 2 FLOPs. BN inference = 4 scalar ops/element
(unfused upper estimate). Sigmoid = 4 scalar ops/element including exp
counted as one; hardware cost of exp is not represented by this convention.
Global mean = N-1 additions + one division per channel. Memory ops ignored.
"""

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from models import LightNet
from models.lightnet import Add, Multiply


def profile_operations(model, input_shape=(1, 3, 32, 32)):
    counts = {"macs": 0, "other_flops": 0}
    handles = []
    supported = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.Sigmoid,
                 nn.AdaptiveAvgPool2d, nn.Identity, Add, Multiply)

    def hook(module, inputs, output):
        elements = output.numel()
        if isinstance(module, nn.Conv2d):
            counts["macs"] += elements * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            counts["other_flops"] += elements if module.bias is not None else 0
        elif isinstance(module, nn.Linear):
            counts["macs"] += elements * module.in_features
            counts["other_flops"] += elements if module.bias is not None else 0
        elif isinstance(module, nn.BatchNorm2d):
            counts["other_flops"] += 4 * elements
        elif isinstance(module, (nn.ReLU, Add, Multiply)):
            counts["other_flops"] += elements
        elif isinstance(module, nn.Sigmoid):
            counts["other_flops"] += 4 * elements
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            if output.shape[-2:] != (1, 1):
                raise ValueError("Only global average pooling is supported")
            counts["other_flops"] += inputs[0].numel()

    for module in model.modules():
        if not list(module.children()):
            if not isinstance(module, supported):
                raise ValueError(f"Unsupported leaf operation: {type(module).__name__}")
            handles.append(module.register_forward_hook(hook))
    training = model.training
    device = next(model.parameters()).device
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(input_shape, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = 2 * counts["macs"] + counts["other_flops"]
    return {
        "input_shape": list(input_shape), "learnable_parameters": parameters,
        **counts, "flops": flops,
        "within_limits": parameters <= 6_000_000 and flops < 1_000_000_000,
        "convention": "Conv/Linear: 2 FLOPs/MAC; BN: 4/element; ReLU/add/mul: 1; sigmoid: 4; GAP: N/channel. No memory costs; unfused inference.",
    }


def benchmark(model, batch_size=128, steps=30, warmup=10):
    if batch_size <= 0 or steps <= 0 or warmup < 0:
        raise ValueError("Invalid benchmark sizes")
    device = next(model.parameters()).device
    training = model.training
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(x)
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            latencies = []
            for _ in range(steps):
                tick = time.perf_counter()
                model(x)
                synchronize()
                latencies.append((time.perf_counter() - tick) * 1000)
            synchronize()
            seconds = time.perf_counter() - start
        return {
            "benchmark_batch_size": batch_size, "precision": "float32",
            "batch_latency_ms": seconds / steps * 1000,
            "batch_latency_median_ms": float(np.median(latencies)),
            "batch_latency_p95_ms": float(np.percentile(latencies, 95)),
            "warmup_steps": warmup, "measurement_steps": steps,
            "images_per_second": steps * batch_size / seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "note": "Forward-only synthetic inputs, not training throughput or accuracy.",
        }
    finally:
        model.train(training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["baseline", "multiscale"], default="multiscale")
    parser.add_argument("--classes", type=int, choices=[10, 100], default=100)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = LightNet(args.classes, args.variant).to(args.device)
    result = {"variant": args.variant, "classes": args.classes, **profile_operations(model)}
    if args.benchmark:
        result.update(benchmark(model, args.batch_size))
    print(json.dumps(result, indent=2))
    if not result["within_limits"]:
        raise SystemExit("Model exceeds final-exam resource constraints")


if __name__ == "__main__":
    main()
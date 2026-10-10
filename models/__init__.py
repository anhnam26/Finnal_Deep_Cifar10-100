from .scalegate import ScaleGateNet

# Keep the existing training/evaluation entrypoint interface, not old weights.
LightNet = ScaleGateNet
__all__ = ["ScaleGateNet", "LightNet"]
from .stagefusion import StageFusionNet

# Keep the existing training/evaluation entrypoint interface, not old weights.
LightNet = StageFusionNet
__all__ = ["StageFusionNet", "LightNet"]
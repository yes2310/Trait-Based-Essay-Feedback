"""Loss modules for PromptAES2."""

from .combined import CombinedLoss
from .multitask import MultiTaskLoss

__all__ = ["CombinedLoss", "MultiTaskLoss"]

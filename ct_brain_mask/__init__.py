"""CT Brain Mask — HU-threshold brain segmentation for CT perfusion imaging."""

from .mask import create_brain_mask, create_brain_mask_4d

__version__ = "0.1.0"
__all__ = ["create_brain_mask", "create_brain_mask_4d"]

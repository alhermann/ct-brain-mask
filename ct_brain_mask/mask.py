"""
CT Brain Mask — HU-threshold-based brain segmentation for CT and dynamic CT imaging.

Algorithm:
  1. Threshold baseline CT at brain parenchyma HU range (default 20-80 HU)
     - Excludes air (<0 HU), fat/CSF (<20 HU), bone/skull (>80 HU)
  2. Fill holes (ventricles, sulci)
  3. Keep only the largest connected component (the brain)
  4. Final hole fill after component selection

Dependencies: numpy, scipy.ndimage
"""

import numpy as np
from scipy import ndimage


def create_brain_mask(ct_baseline_2d, hu_min=20, hu_max=80,
                      median_size=None, opening=False, verbose=True):
    """
    Create a binary brain mask from a 2D baseline CT image in Hounsfield Units.

    Parameters
    ----------
    ct_baseline_2d : np.ndarray, shape (H, W)
        Baseline (pre-contrast) CT image in HU.
    hu_min : float
        Lower HU threshold. Default 20 excludes air, fat, and CSF.
    hu_max : float
        Upper HU threshold. Default 80 excludes skull and bone (typically >200 HU).
        The 80 HU cutoff naturally separates parenchyma from skull without erosion.
    median_size : int or None
        If set, apply a median filter of this kernel size before thresholding.
        Reduces noise and salt-and-pepper artifacts. Typical values: 3 or 5.
    opening : bool or int
        If truthy, apply binary opening after thresholding to remove small
        fragments and artifacts. ``True`` uses 1 iteration; an int specifies
        the number of iterations.
    verbose : bool
        Print mask statistics.

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype bool
        Binary brain mask.
    """
    img = ct_baseline_2d

    # Optional median filter for noise reduction
    if median_size is not None:
        img = ndimage.median_filter(img, size=median_size)

    # Threshold at brain parenchyma HU range
    mask = (img > hu_min) & (img < hu_max)

    # Optional morphological opening to clean fragments/artifacts
    if opening:
        iterations = opening if isinstance(opening, int) and opening > 1 else 1
        mask = ndimage.binary_opening(mask, iterations=iterations)

    # Fill holes (ventricles, internal CSF spaces)
    mask = ndimage.binary_fill_holes(mask)

    # Keep only the largest connected component (the brain)
    labeled, n_features = ndimage.label(mask)
    if n_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, n_features + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

    # Final fill to catch any remaining holes after component selection
    mask = ndimage.binary_fill_holes(mask)

    if verbose:
        print(f"  Brain mask: {mask.sum():,} voxels "
              f"({mask.sum() * 100 / mask.size:.1f}% of "
              f"{mask.shape[0]}x{mask.shape[1]}) [HU {hu_min}-{hu_max}]")

    return mask.astype(bool)


def create_brain_mask_4d(volume_4d, slice_idx, hu_min=20, hu_max=80,
                         n_baseline=3, median_size=None, opening=False,
                         verbose=True):
    """
    Create a brain mask from a 4D dynamic CT volume.

    Averages the first ``n_baseline`` frames (pre-contrast) to get a stable
    baseline image, then applies HU thresholding.

    Parameters
    ----------
    volume_4d : np.ndarray, shape (S, H, W, T)
        4D dynamic CT volume in HU (slices x height x width x time).
    slice_idx : int
        Which slice to mask.
    hu_min : float
        Lower HU threshold (default 20).
    hu_max : float
        Upper HU threshold (default 80).
    n_baseline : int
        Number of initial frames to average for baseline (default 3).
    median_size : int or None
        If set, apply a median filter before thresholding (see ``create_brain_mask``).
    opening : bool or int
        If truthy, apply binary opening after thresholding (see ``create_brain_mask``).
    verbose : bool
        Print mask statistics.

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype bool
        Binary brain mask.
    """
    ct_baseline = volume_4d[slice_idx, :, :, :n_baseline].mean(axis=-1)
    return create_brain_mask(ct_baseline, hu_min=hu_min, hu_max=hu_max,
                             median_size=median_size, opening=opening,
                             verbose=verbose)

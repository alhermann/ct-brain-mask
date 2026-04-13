"""
CT Brain Mask — brain segmentation for CT and dynamic CT imaging.

Two algorithms available:

**v1 (``create_brain_mask``)** — Simple HU thresholding:
  1. Threshold at brain HU range (default 20-80)
  2. Fill holes, keep largest component

**v2 (``create_brain_mask_robust``)** — Artifact-resistant (recommended):
  1. Wider HU threshold [10-90] to catch beam-hardening-affected tissue
  2. Morphological closing to bridge artifact gaps near skull
  3. Gaussian boundary smoothing for clean edges
  4. Convex hull with HU constraint to fill concavities
  5. Light erosion (1px)

  v2 fixes "bite" artifacts in superior slices caused by beam hardening
  near the skull. Validated on Asklepios CTP data: recovers ~3% more
  brain tissue, eliminates concavity artifacts in S11-S15.

Dependencies: numpy, scipy.ndimage
Optional for v2: scipy.spatial.ConvexHull, matplotlib.path.Path
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


def create_brain_mask_robust(ct_baseline_2d, hu_min=10, hu_max=90,
                             closing_iters=6, smooth_sigma=2.5,
                             erode_px=1, use_convex_hull=True,
                             hull_hu_max=120, verbose=True):
    """
    Create an artifact-resistant brain mask from a 2D baseline CT image (v2).

    Fixes beam hardening "bite" artifacts near the skull using morphological
    closing, Gaussian smoothing, and HU-constrained convex hull.

    Parameters
    ----------
    ct_baseline_2d : np.ndarray, shape (H, W)
        Baseline (pre-contrast) CT image in HU.
    hu_min : float
        Lower HU threshold (default 10, wider than v1 to catch artifacts).
    hu_max : float
        Upper HU threshold (default 90).
    closing_iters : int
        Iterations of morphological closing to bridge gaps (default 6).
    smooth_sigma : float
        Gaussian sigma for boundary smoothing (default 2.5).
    erode_px : int
        Final erosion iterations (default 1, lighter than v1's 2).
    use_convex_hull : bool
        Apply convex hull to fill concavities (default True).
    hull_hu_max : float
        Maximum HU for convex hull voxels (default 120, excludes bone).
    verbose : bool
        Print mask statistics.

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype bool
        Binary brain mask.
    """
    H, W = ct_baseline_2d.shape

    # Step 1: Wide HU threshold
    mask = (ct_baseline_2d > hu_min) & (ct_baseline_2d < hu_max)

    # Step 2: Fill holes + largest component
    mask = ndimage.binary_fill_holes(mask)
    labeled, nc = ndimage.label(mask)
    if nc > 1:
        sizes = ndimage.sum(mask, labeled, range(1, nc + 1))
        mask = labeled == (np.argmax(sizes) + 1)

    # Step 3: Morphological closing — bridge beam hardening gaps
    if closing_iters > 0:
        mask = ndimage.binary_closing(mask, iterations=closing_iters)
        mask = ndimage.binary_fill_holes(mask)

    # Step 4: Gaussian boundary smoothing
    if smooth_sigma > 0:
        sm = ndimage.gaussian_filter(mask.astype(np.float32), sigma=smooth_sigma)
        mask = sm > 0.5
        mask = ndimage.binary_fill_holes(mask)

    # Keep largest component
    labeled, nc = ndimage.label(mask)
    if nc > 1:
        sizes = ndimage.sum(mask, labeled, range(1, nc + 1))
        mask = labeled == (np.argmax(sizes) + 1)

    # Step 5: Convex hull with HU constraint
    if use_convex_hull:
        try:
            from scipy.spatial import ConvexHull
            from matplotlib.path import Path

            ys, xs = np.where(mask)
            if len(ys) > 10:
                pts = np.column_stack([xs, ys])
                hull = ConvexHull(pts)
                hp = Path(pts[hull.vertices])
                yy, xx = np.mgrid[0:H, 0:W]
                gp = np.column_stack([xx.ravel(), yy.ravel()])
                hull_mask = hp.contains_points(gp).reshape(H, W)
                # Only add hull voxels with brain-like HU (exclude bone, air)
                hull_brain = hull_mask & (ct_baseline_2d > 0) & (ct_baseline_2d < hull_hu_max)
                mask = mask | hull_brain
        except (ImportError, Exception):
            pass  # ConvexHull not available or failed — skip

    # Final cleanup
    mask = ndimage.binary_fill_holes(mask)
    labeled, nc = ndimage.label(mask)
    if nc > 1:
        sizes = ndimage.sum(mask, labeled, range(1, nc + 1))
        mask = labeled == (np.argmax(sizes) + 1)

    # Step 6: Light erosion
    if erode_px > 0:
        mask = ndimage.binary_erosion(mask, iterations=erode_px)

    if verbose:
        print(f"  Brain mask (v2): {mask.sum():,} voxels "
              f"({mask.sum() * 100 / mask.size:.1f}% of "
              f"{H}x{W}) [HU {hu_min}-{hu_max}, closing={closing_iters}, "
              f"hull={'on' if use_convex_hull else 'off'}]")

    return mask.astype(bool)


def create_brain_mask_volume(volume_4d, n_baseline=3, robust=True, verbose=True,
                              **kwargs):
    """
    Create a 3D brain mask for an entire 4D dynamic CT volume.

    Applies mask creation to each slice independently using the
    baseline (pre-contrast) average.

    Parameters
    ----------
    volume_4d : np.ndarray, shape (S, H, W, T)
        4D dynamic CT volume in HU.
    n_baseline : int
        Number of initial frames to average (default 3).
    robust : bool
        If True, use ``create_brain_mask_robust`` (v2, recommended).
        If False, use ``create_brain_mask`` (v1, simple).
    verbose : bool
        Print per-slice statistics.
    **kwargs
        Additional arguments passed to the mask function.

    Returns
    -------
    mask_3d : np.ndarray, shape (S, H, W), dtype bool
        3D brain mask.
    """
    S, H, W, T = volume_4d.shape
    baseline = volume_4d[:, :, :, :n_baseline].mean(axis=-1)

    mask_fn = create_brain_mask_robust if robust else create_brain_mask
    mask_3d = np.zeros((S, H, W), dtype=bool)

    for si in range(S):
        mask_3d[si] = mask_fn(baseline[si], verbose=False, **kwargs)

    total = mask_3d.sum()
    version = "v2 robust" if robust else "v1 simple"
    if verbose:
        print(f"  Brain mask ({version}): {total:,} voxels "
              f"({total * 100 / (S * H * W):.1f}% of {S}x{H}x{W})")
    return mask_3d


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

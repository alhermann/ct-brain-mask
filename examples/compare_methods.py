#!/usr/bin/env python3
"""
Compare brain masking methods on a dynamic CT DICOM dataset.

Generates a 4-panel figure:
  1. Baseline CT image
  2. Our mask (HU 20-80) — parenchyma-only, excludes skull
  3. Lenient mask (HU 20-1300) — includes skull/bone
  4. Naive mask (> 0) — includes everything non-air

Usage:
    python compare_methods.py --dicom_dir /path/to/DICOM/folder --slice_idx 8
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pydicom

# Add parent directory to path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ct_brain_mask import create_brain_mask


def load_dicom_slice(dicom_dir, slice_idx=8, n_baseline=3):
    """Load a single baseline CT slice from a DICOM dynamic CT directory."""
    files = glob.glob(os.path.join(dicom_dir, '*'))
    if not files:
        raise FileNotFoundError(f"No files in {dicom_dir}")

    records = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            sl = float(ds.SliceLocation)
            hu = (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
                  + float(ds.RescaleIntercept))
            records.append((sl, hu))
        except Exception:
            continue

    # Group by slice location
    slice_locs = sorted(set(r[0] for r in records))
    n_slices = len(slice_locs)
    sl_to_idx = {sl: i for i, sl in enumerate(slice_locs)}

    if slice_idx >= n_slices:
        slice_idx = n_slices // 2
        print(f"  Adjusted slice_idx to {slice_idx} (only {n_slices} slices)")

    target_sl = slice_locs[slice_idx]

    # Collect frames for this slice
    frames = [r[1] for r in records if r[0] == target_sl]
    baseline = np.mean(frames[:n_baseline], axis=0)

    print(f"  Loaded slice {slice_idx}/{n_slices} "
          f"({baseline.shape[0]}x{baseline.shape[1]}), "
          f"{len(frames)} phases, baseline HU range: "
          f"[{baseline.min():.0f}, {baseline.max():.0f}]")
    return baseline


def main():
    parser = argparse.ArgumentParser(description="Compare brain mask methods")
    parser.add_argument('--dicom_dir', required=True, help="DICOM dynamic CT directory")
    parser.add_argument('--slice_idx', type=int, default=8, help="Slice index")
    parser.add_argument('--output', default='examples/mask_comparison.png',
                        help="Output image path")
    args = parser.parse_args()

    print("Loading DICOM data...")
    baseline = load_dicom_slice(args.dicom_dir, args.slice_idx)

    # --- Generate three masks ---
    print("\nMethod 1: Our mask (HU 20-80) — parenchyma-only")
    mask_ours = create_brain_mask(baseline, hu_min=20, hu_max=80)

    print("Method 2: Lenient mask (HU 20-1300) — includes skull")
    mask_lenient = create_brain_mask(baseline, hu_min=20, hu_max=1300)

    print("Method 3: Naive mask (> 0) — everything non-air")
    mask_naive = create_brain_mask(baseline, hu_min=0, hu_max=9999)

    # --- Plot comparison ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    ct_vmin, ct_vmax = 0, 80  # brain window

    # Panel 1: Baseline CT
    axes[0].imshow(baseline, cmap='gray', vmin=ct_vmin, vmax=ct_vmax)
    axes[0].set_title(f'Baseline CT (slice {args.slice_idx})', fontsize=12)
    axes[0].axis('off')

    # Panel 2: Our mask (HU 20-80)
    axes[1].imshow(baseline, cmap='gray', vmin=ct_vmin, vmax=ct_vmax)
    axes[1].imshow(np.ma.masked_where(~mask_ours, mask_ours.astype(float)),
                   cmap='Reds', alpha=0.4, vmin=0, vmax=1)
    axes[1].set_title(f'HU [20, 80] — {mask_ours.sum():,} vox', fontsize=12,
                      color='darkred')
    axes[1].axis('off')

    # Panel 3: Lenient mask (HU 20-1300)
    axes[2].imshow(baseline, cmap='gray', vmin=ct_vmin, vmax=ct_vmax)
    axes[2].imshow(np.ma.masked_where(~mask_lenient, mask_lenient.astype(float)),
                   cmap='Blues', alpha=0.4, vmin=0, vmax=1)
    axes[2].set_title(f'HU [20, 1300] — {mask_lenient.sum():,} vox', fontsize=12,
                      color='darkblue')
    axes[2].axis('off')

    # Panel 4: Naive mask (> 0)
    axes[3].imshow(baseline, cmap='gray', vmin=ct_vmin, vmax=ct_vmax)
    axes[3].imshow(np.ma.masked_where(~mask_naive, mask_naive.astype(float)),
                   cmap='Greens', alpha=0.4, vmin=0, vmax=1)
    axes[3].set_title(f'HU > 0 — {mask_naive.sum():,} vox', fontsize=12,
                      color='darkgreen')
    axes[3].axis('off')

    fig.suptitle('CT Brain Mask Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to {args.output}")
    plt.close()


if __name__ == '__main__':
    main()

# ct-brain-mask

Simple, robust brain segmentation for non-contrast CT and dynamic CT images using Hounsfield Unit thresholding and morphological operations. No deep learning, no training data — just numpy and scipy.

## Mask Comparison

![Brain mask comparison](examples/brain_mask_demo.png)

*Public CT scan from [niivue-images](https://github.com/neurolabusc/niivue-images) (CT_Philips, slice 157).*

| Method | Voxels | Coverage | Skull included? |
|--------|--------|----------|-----------------|
| **HU [20, 80] (ours)** | 19,218 | 44.8% | No |
| HU [20, 1300] | 25,270 | 58.9% | Yes |
| HU > 0 | 25,478 | 59.4% | Yes |

The [20, 80] HU window cleanly isolates brain parenchyma. Broader thresholds include skull and bone, which is problematic for perfusion analysis.

## Algorithm

1. **HU threshold** the baseline CT image at [20, 80] HU
   - Excludes air (< 0 HU), fat/CSF (< 20 HU), bone/skull (> 80 HU)
   - The 80 HU upper bound naturally separates parenchyma from skull — no morphological erosion needed
2. **Binary hole filling** — recaptures ventricles, sulci, and internal CSF spaces
3. **Largest connected component** — removes isolated fragments outside the brain
4. **Final hole fill** — closes any remaining gaps after component selection

## Why HU 20–80?

| Tissue | Typical HU |
|--------|------------|
| Air | −1000 |
| Fat | −100 to −50 |
| CSF | 0–15 |
| **White matter** | **20–35** |
| **Gray matter** | **30–45** |
| Blood | 30–45 |
| Soft tissue | 40–80 |
| Bone / skull | 200–3000 |

The [20, 80] window captures all brain parenchyma and blood while naturally excluding skull (typically > 200 HU). CSF-filled spaces (0–15 HU) are excluded by the threshold but recovered by the hole-filling step. This avoids the need for morphological erosion or atlas-based skull stripping.

## Installation

From [PyPI](https://pypi.org/project/ct-brain-mask/) (once published):

```bash
pip install ct-brain-mask
```

From [TestPyPI](https://test.pypi.org/project/ct-brain-mask/) (current):

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ct-brain-mask
```

From source:

```bash
git clone https://github.com/alhermann/ct-brain-mask.git
cd ct-brain-mask
pip install -e .
```

## Usage

```python
from ct_brain_mask import create_brain_mask

# From a 2D baseline CT image (H, W) in Hounsfield Units
mask = create_brain_mask(ct_baseline_2d)
# Brain mask: 89,527 voxels (34.2% of 512x512) [HU 20-80]

# Custom thresholds
mask = create_brain_mask(ct_baseline_2d, hu_min=10, hu_max=100)

# From a 4D dynamic CT volume (slices, H, W, time)
from ct_brain_mask import create_brain_mask_4d
mask = create_brain_mask_4d(volume_4d, slice_idx=8, n_baseline=3)
```

### API

**`create_brain_mask(ct_baseline_2d, hu_min=20, hu_max=80, verbose=True)`**

Create a binary brain mask from a 2D CT image in Hounsfield Units.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ct_baseline_2d` | ndarray (H, W) | required | Baseline CT image in HU |
| `hu_min` | float | 20 | Lower HU threshold |
| `hu_max` | float | 80 | Upper HU threshold |
| `verbose` | bool | True | Print mask statistics |

Returns: `ndarray (H, W)`, dtype `bool`

**`create_brain_mask_4d(volume_4d, slice_idx, hu_min=20, hu_max=80, n_baseline=3, verbose=True)`**

Convenience wrapper for 4D dynamic CT volumes. Averages the first `n_baseline` frames (pre-contrast) to compute a stable baseline, then calls `create_brain_mask`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `volume_4d` | ndarray (S, H, W, T) | required | 4D dynamic CT volume in HU |
| `slice_idx` | int | required | Slice index to mask |
| `n_baseline` | int | 3 | Pre-contrast frames to average |

Returns: `ndarray (H, W)`, dtype `bool`

## Dependencies

- Python >= 3.8
- numpy
- scipy

Optional (for examples): matplotlib, pydicom, nibabel

## License

MIT

"""Unit tests for ct_brain_mask."""

import os

import numpy as np
import pytest

from ct_brain_mask import create_brain_mask, create_brain_mask_4d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_circle(shape=(512, 512), center=None, radius=100, value=30.0,
                 background=-1000.0):
    """Create a synthetic CT image with a circular 'brain' region."""
    h, w = shape
    if center is None:
        center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    img = np.full(shape, background, dtype=np.float64)
    img[dist <= radius] = value
    return img


def _make_brain_with_skull(shape=(512, 512), brain_radius=80,
                           skull_radius=100, brain_hu=35.0, skull_hu=800.0):
    """Create a synthetic CT with brain parenchyma surrounded by skull."""
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    img = np.full(shape, -1000.0, dtype=np.float64)
    # Skull ring
    img[dist <= skull_radius] = skull_hu
    # Brain inside skull
    img[dist <= brain_radius] = brain_hu
    return img


# ---------------------------------------------------------------------------
# Tests for create_brain_mask
# ---------------------------------------------------------------------------

class TestCreateBrainMask:
    """Tests for the 2D brain mask function."""

    def test_basic_circle(self):
        """A circle at 30 HU should be detected as brain."""
        img = _make_circle(value=30.0)
        mask = create_brain_mask(img, verbose=False)
        assert mask.dtype == bool
        assert mask.shape == img.shape
        # The circle should have many True voxels
        assert mask.sum() > 1000

    def test_output_shape_matches_input(self):
        """Output shape must equal input shape for various sizes."""
        for shape in [(64, 64), (128, 256), (512, 512)]:
            img = _make_circle(shape=shape, radius=min(shape) // 4)
            mask = create_brain_mask(img, verbose=False)
            assert mask.shape == shape

    def test_excludes_air(self):
        """An image with only air (-1000 HU) should produce an empty mask."""
        img = np.full((128, 128), -1000.0)
        mask = create_brain_mask(img, verbose=False)
        assert mask.sum() == 0

    def test_excludes_bone(self):
        """An image with only bone (800 HU) should produce an empty mask."""
        img = np.full((128, 128), 800.0)
        mask = create_brain_mask(img, verbose=False)
        assert mask.sum() == 0

    def test_excludes_csf(self):
        """An image with only CSF (10 HU) should produce an empty mask."""
        img = np.full((128, 128), 10.0)
        mask = create_brain_mask(img, verbose=False)
        assert mask.sum() == 0

    def test_skull_exclusion(self):
        """Skull (high HU) should not appear in the mask."""
        img = _make_brain_with_skull(brain_radius=80, skull_radius=100,
                                     brain_hu=35.0, skull_hu=800.0)
        mask = create_brain_mask(img, verbose=False)
        # Brain area: pi * 80^2 ≈ 20106
        # Skull ring area: pi * (100^2 - 80^2) ≈ 11310
        # Mask should be close to brain area, not including skull ring
        brain_area = np.pi * 80 ** 2
        skull_ring_area = np.pi * (100 ** 2 - 80 ** 2)
        # Mask should not include skull — allow some tolerance for discretization
        assert mask.sum() < brain_area + skull_ring_area * 0.1

    def test_hole_filling(self):
        """Internal holes (e.g., ventricles at CSF HU) should be filled."""
        img = _make_circle(value=35.0, radius=80)
        # Punch a hole in the center (simulating ventricle at CSF HU)
        cy, cx = 256, 256
        y, x = np.ogrid[:512, :512]
        ventricle = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) <= 15
        img[ventricle] = 5.0  # CSF HU
        mask = create_brain_mask(img, verbose=False)
        # The ventricle should be filled in
        assert mask[cy, cx] == True

    def test_largest_component(self):
        """Only the largest connected component should be kept."""
        img = np.full((256, 256), -1000.0)
        # Big circle
        y, x = np.ogrid[:256, :256]
        big = np.sqrt((y - 128) ** 2 + (x - 128) ** 2) <= 50
        img[big] = 35.0
        # Small circle far away
        small = np.sqrt((y - 20) ** 2 + (x - 20) ** 2) <= 8
        img[small] = 35.0
        mask = create_brain_mask(img, verbose=False)
        # Small component should be removed
        assert mask[20, 20] == False
        # Big component should remain
        assert mask[128, 128] == True

    def test_custom_thresholds(self):
        """Custom hu_min/hu_max should be respected."""
        img = _make_circle(value=50.0)
        # With default thresholds [20, 80], value=50 is in range
        mask1 = create_brain_mask(img, hu_min=20, hu_max=80, verbose=False)
        assert mask1.sum() > 0
        # With narrow thresholds [60, 80], value=50 is out of range
        mask2 = create_brain_mask(img, hu_min=60, hu_max=80, verbose=False)
        assert mask2.sum() == 0

    def test_verbose_output(self, capsys):
        """Verbose mode should print mask statistics."""
        img = _make_circle(value=35.0)
        create_brain_mask(img, verbose=True)
        captured = capsys.readouterr()
        assert "Brain mask:" in captured.out
        assert "voxels" in captured.out

    def test_verbose_off(self, capsys):
        """verbose=False should print nothing."""
        img = _make_circle(value=35.0)
        create_brain_mask(img, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_uniform_brain_hu(self):
        """A uniform image within HU range should produce a full mask."""
        img = np.full((64, 64), 40.0)
        mask = create_brain_mask(img, verbose=False)
        assert mask.all()

    def test_boundary_values(self):
        """Values exactly at hu_min and hu_max boundaries."""
        # Exactly at hu_min=20: should be excluded (strict >)
        img_min = np.full((64, 64), 20.0)
        mask_min = create_brain_mask(img_min, hu_min=20, hu_max=80, verbose=False)
        assert mask_min.sum() == 0

        # Exactly at hu_max=80: should be excluded (strict <)
        img_max = np.full((64, 64), 80.0)
        mask_max = create_brain_mask(img_max, hu_min=20, hu_max=80, verbose=False)
        assert mask_max.sum() == 0

        # Just inside the range
        img_in = np.full((64, 64), 50.0)
        mask_in = create_brain_mask(img_in, hu_min=20, hu_max=80, verbose=False)
        assert mask_in.sum() == 64 * 64

    def test_empty_image(self):
        """Very small image edge case."""
        img = np.array([[35.0]])
        mask = create_brain_mask(img, verbose=False)
        assert mask.shape == (1, 1)
        assert mask[0, 0] == True

    def test_returns_bool_dtype(self):
        """Mask should always be boolean dtype."""
        img = _make_circle(value=40.0)
        mask = create_brain_mask(img, verbose=False)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Tests for create_brain_mask_4d
# ---------------------------------------------------------------------------

class TestCreateBrainMask4D:
    """Tests for the 4D convenience wrapper."""

    def _make_volume(self, shape=(5, 128, 128, 10), brain_value=35.0):
        """Create a synthetic 4D dynamic CT volume."""
        vol = np.full(shape, -1000.0)
        s, h, w, t = shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        brain = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) <= 40
        for si in range(s):
            for ti in range(t):
                vol[si, brain, ti] = brain_value
        return vol

    def test_basic_4d(self):
        """Basic 4D masking should work."""
        vol = self._make_volume()
        mask = create_brain_mask_4d(vol, slice_idx=2, verbose=False)
        assert mask.dtype == bool
        assert mask.shape == (128, 128)
        assert mask.sum() > 0

    def test_slice_idx(self):
        """Different slice indices should work."""
        vol = self._make_volume()
        for si in range(5):
            mask = create_brain_mask_4d(vol, slice_idx=si, verbose=False)
            assert mask.shape == (128, 128)

    def test_n_baseline(self):
        """n_baseline parameter should control averaging."""
        vol = self._make_volume(shape=(2, 64, 64, 10))
        # With different n_baseline, result should still be valid
        mask1 = create_brain_mask_4d(vol, slice_idx=0, n_baseline=1, verbose=False)
        mask5 = create_brain_mask_4d(vol, slice_idx=0, n_baseline=5, verbose=False)
        assert mask1.shape == (64, 64)
        assert mask5.shape == (64, 64)

    def test_baseline_averaging_reduces_noise(self):
        """Averaging multiple baselines should produce a cleaner result."""
        rng = np.random.RandomState(42)
        vol = np.full((1, 128, 128, 20), -1000.0)
        y, x = np.ogrid[:128, :128]
        brain = np.sqrt((y - 64) ** 2 + (x - 64) ** 2) <= 40
        for t in range(20):
            # Add brain signal with noise
            noise = rng.normal(0, 8, (128, 128))
            slice_img = np.full((128, 128), -1000.0)
            slice_img[brain] = 35.0
            slice_img += noise
            vol[0, :, :, t] = slice_img
        # More baseline frames should give a valid mask
        mask = create_brain_mask_4d(vol, slice_idx=0, n_baseline=10, verbose=False)
        assert mask.sum() > 0

    def test_custom_thresholds_4d(self):
        """Custom HU thresholds should propagate to the 2D function."""
        vol = self._make_volume(brain_value=50.0)
        mask_default = create_brain_mask_4d(vol, slice_idx=0, verbose=False)
        mask_narrow = create_brain_mask_4d(vol, slice_idx=0, hu_min=60,
                                           hu_max=80, verbose=False)
        assert mask_default.sum() > 0
        assert mask_narrow.sum() == 0


# ---------------------------------------------------------------------------
# Tests with real DICOM data (skipped if data not available)
# ---------------------------------------------------------------------------

DICOM_DIR = "/Users/hermann/Desktop/SPPINNs/DICOM/0000413A"


def _load_dicom_slice(dicom_dir):
    """Load a single DICOM file and convert to HU."""
    try:
        import pydicom
    except ImportError:
        return None

    files = sorted([f for f in os.listdir(dicom_dir) if not f.startswith('.')])
    if not files:
        return None

    ds = pydicom.dcmread(os.path.join(dicom_dir, files[0]))
    pixel_array = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    hu_image = pixel_array * slope + intercept
    return hu_image


def _load_dicom_volume(dicom_dir, max_slices=5):
    """Load multiple DICOM files as a pseudo-4D volume for testing."""
    try:
        import pydicom
    except ImportError:
        return None

    files = sorted([f for f in os.listdir(dicom_dir) if not f.startswith('.')])
    if not files:
        return None

    slices = []
    for f in files[:max_slices]:
        ds = pydicom.dcmread(os.path.join(dicom_dir, f))
        pixel_array = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        slices.append(pixel_array * slope + intercept)

    # Stack as (1, H, W, T) - single slice, multiple "timepoints"
    arr = np.stack(slices, axis=-1)  # (H, W, T)
    return arr[np.newaxis, ...]  # (1, H, W, T)


@pytest.fixture
def dicom_image():
    """Load a single DICOM slice in HU."""
    if not os.path.isdir(DICOM_DIR):
        pytest.skip("DICOM data not available")
    img = _load_dicom_slice(DICOM_DIR)
    if img is None:
        pytest.skip("Could not load DICOM data")
    return img


@pytest.fixture
def dicom_volume():
    """Load DICOM files as a pseudo-4D volume."""
    if not os.path.isdir(DICOM_DIR):
        pytest.skip("DICOM data not available")
    vol = _load_dicom_volume(DICOM_DIR)
    if vol is None:
        pytest.skip("Could not load DICOM volume")
    return vol


class TestWithDICOM:
    """Tests using real DICOM data."""

    def test_dicom_mask_has_content(self, dicom_image):
        """Mask from real DICOM should have brain voxels."""
        mask = create_brain_mask(dicom_image, verbose=False)
        assert mask.dtype == bool
        assert mask.shape == dicom_image.shape
        assert mask.sum() > 0

    def test_dicom_mask_reasonable_coverage(self, dicom_image):
        """Brain should be a reasonable fraction of the image (5-60%)."""
        mask = create_brain_mask(dicom_image, verbose=False)
        coverage = mask.sum() / mask.size
        assert 0.05 < coverage < 0.60, f"Coverage {coverage:.1%} seems unreasonable"

    def test_dicom_mask_excludes_skull(self, dicom_image):
        """Mask at [20,80] should have fewer voxels than at [20,1300]."""
        mask_brain = create_brain_mask(dicom_image, hu_min=20, hu_max=80,
                                       verbose=False)
        mask_with_skull = create_brain_mask(dicom_image, hu_min=20, hu_max=1300,
                                            verbose=False)
        assert mask_brain.sum() < mask_with_skull.sum()

    def test_dicom_mask_is_contiguous(self, dicom_image):
        """Brain mask should be a single connected component."""
        mask = create_brain_mask(dicom_image, verbose=False)
        from scipy import ndimage
        labeled, n = ndimage.label(mask)
        assert n == 1, f"Expected 1 component, got {n}"

    def test_dicom_4d_works(self, dicom_volume):
        """4D wrapper should work with real DICOM data."""
        mask = create_brain_mask_4d(dicom_volume, slice_idx=0, n_baseline=3,
                                    verbose=False)
        assert mask.dtype == bool
        assert mask.sum() > 0

    def test_dicom_verbose_output(self, dicom_image, capsys):
        """Verbose output should show realistic stats for real data."""
        create_brain_mask(dicom_image, verbose=True)
        captured = capsys.readouterr()
        assert "Brain mask:" in captured.out
        assert "512x512" in captured.out

    def test_dicom_hu_range_sanity(self, dicom_image):
        """DICOM HU values should span a reasonable range for CT."""
        assert dicom_image.min() < 0, "CT should have negative HU (air)"
        assert dicom_image.max() > 80, "CT should have values above 80 HU (bone)"

    def test_dicom_reproducibility(self, dicom_image):
        """Same input should always produce the same mask."""
        mask1 = create_brain_mask(dicom_image, verbose=False)
        mask2 = create_brain_mask(dicom_image, verbose=False)
        np.testing.assert_array_equal(mask1, mask2)


# ---------------------------------------------------------------------------
# Tests with ISLES 2024 public NIfTI data (skipped if data not available)
# ---------------------------------------------------------------------------

ISLES24_DIR = os.path.join(os.path.dirname(__file__),
                           '..', 'examples', 'isles24_sample')


def _have_isles24():
    """Check if ISLES24 sample NIfTI files are available."""
    return os.path.isfile(os.path.join(ISLES24_DIR, 'cbf.nii.gz'))


@pytest.fixture
def isles24_cbf():
    """Load ISLES24 CBF map, slice 37."""
    if not _have_isles24():
        pytest.skip("ISLES24 sample data not available")
    import nibabel as nib
    data = nib.load(os.path.join(ISLES24_DIR, 'cbf.nii.gz')).get_fdata()
    return data[:, :, 37]


@pytest.fixture
def isles24_tmax():
    """Load ISLES24 Tmax map, slice 37."""
    if not _have_isles24():
        pytest.skip("ISLES24 sample data not available")
    import nibabel as nib
    data = nib.load(os.path.join(ISLES24_DIR, 'tmax.nii.gz')).get_fdata()
    return data[:, :, 37]


@pytest.fixture
def isles24_lesion():
    """Load ISLES24 lesion mask, slice 37."""
    if not _have_isles24():
        pytest.skip("ISLES24 sample data not available")
    import nibabel as nib
    data = nib.load(os.path.join(ISLES24_DIR, 'lesion.nii.gz')).get_fdata()
    return data[:, :, 37]


class TestWithISLES24:
    """Tests using ISLES 2024 public perfusion data."""

    def test_isles24_files_loadable(self):
        """All ISLES24 NIfTI files should be loadable."""
        if not _have_isles24():
            pytest.skip("ISLES24 sample data not available")
        import nibabel as nib
        for name in ['cbf', 'cbv', 'tmax', 'lesion']:
            img = nib.load(os.path.join(ISLES24_DIR, f'{name}.nii.gz'))
            data = img.get_fdata()
            assert data.ndim == 3
            assert data.shape[0] > 0

    def test_isles24_cbf_has_brain_region(self, isles24_cbf):
        """CBF map should have a non-zero brain region."""
        brain_region = isles24_cbf > 0
        assert brain_region.sum() > 10000, "CBF should show brain perfusion"

    def test_isles24_tmax_has_data(self, isles24_tmax):
        """Tmax map should have a meaningful value range."""
        assert isles24_tmax.max() > 0, "Tmax should have positive values"

    def test_isles24_lesion_is_binary(self, isles24_lesion):
        """Lesion mask should be binary (0 or 1)."""
        unique = np.unique(isles24_lesion)
        assert set(unique).issubset({0.0, 1.0})

    def test_isles24_lesion_has_content(self, isles24_lesion):
        """Lesion mask should have lesion voxels at slice 37."""
        assert isles24_lesion.sum() > 1000

    def test_isles24_brain_mask_from_cbf(self, isles24_cbf):
        """Brain mask derived from CBF should be a single connected region."""
        from scipy import ndimage
        brain = isles24_cbf > 0
        brain = ndimage.binary_fill_holes(brain)
        labeled, n = ndimage.label(brain)
        if n > 1:
            sizes = ndimage.sum(brain, labeled, range(1, n + 1))
            largest = np.argmax(sizes) + 1
            brain = labeled == largest
            brain = ndimage.binary_fill_holes(brain)
        labeled2, n2 = ndimage.label(brain)
        assert n2 == 1, f"Expected 1 component after cleanup, got {n2}"

    def test_isles24_lesion_within_brain(self, isles24_cbf, isles24_lesion):
        """Lesion should be mostly within the brain region."""
        from scipy import ndimage
        brain = isles24_cbf > 0
        brain = ndimage.binary_fill_holes(brain)
        labeled, n = ndimage.label(brain)
        if n > 1:
            sizes = ndimage.sum(brain, labeled, range(1, n + 1))
            largest = np.argmax(sizes) + 1
            brain = labeled == largest
            brain = ndimage.binary_fill_holes(brain)
        lesion_in_brain = (isles24_lesion > 0) & brain
        total_lesion = (isles24_lesion > 0).sum()
        overlap = lesion_in_brain.sum() / total_lesion if total_lesion > 0 else 0
        assert overlap > 0.90, f"Only {overlap:.1%} of lesion is within brain"

    def test_isles24_data_shapes_consistent(self):
        """All ISLES24 maps should have the same spatial dimensions."""
        if not _have_isles24():
            pytest.skip("ISLES24 sample data not available")
        import nibabel as nib
        shapes = []
        for name in ['cbf', 'cbv', 'tmax', 'lesion']:
            data = nib.load(os.path.join(ISLES24_DIR, f'{name}.nii.gz')).get_fdata()
            shapes.append(data.shape)
        assert all(s == shapes[0] for s in shapes), f"Shapes differ: {shapes}"

    def test_isles24_cbf_reasonable_range(self, isles24_cbf):
        """CBF values should be in a physiologically reasonable range."""
        nonzero = isles24_cbf[isles24_cbf > 0]
        assert nonzero.mean() < 200, f"Mean CBF {nonzero.mean():.1f} seems too high"
        assert nonzero.mean() > 1, f"Mean CBF {nonzero.mean():.1f} seems too low"

    def test_isles24_version_import(self):
        """Package should expose __version__."""
        from ct_brain_mask import __version__
        assert __version__
        assert isinstance(__version__, str)

"""Stress tests for ct_brain_mask robustness against noise and artifacts.

Artifact simulation methods mirror those in generate_robustness_demo.py:
- Sinogram-domain (via Radon transform): metal streak artifacts
- Physically accurate image-domain: quantum noise, beam hardening, ring artifacts
- Multi-step image-domain approximation: patient motion, photon starvation
"""

import numpy as np
import pytest
from scipy import ndimage
from skimage.transform import radon, iradon

from ct_brain_mask import create_brain_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phantom(shape=(256, 256), brain_radius=80, brain_hu=35.0,
                  skull_radius=100, skull_hu=800.0, background=-1000.0):
    """Create a synthetic CT phantom with brain + skull ring."""
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    img = np.full(shape, background, dtype=np.float64)
    img[dist <= skull_radius] = skull_hu
    img[dist <= brain_radius] = brain_hu
    return img


def _ground_truth(shape=(256, 256), brain_radius=80):
    """Binary ground-truth mask for the phantom brain region."""
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return dist <= brain_radius


def _dice(mask, gt):
    """Dice similarity coefficient between two boolean masks."""
    intersection = np.logical_and(mask, gt).sum()
    total = mask.sum() + gt.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def _add_gaussian_noise(img, sigma, rng):
    """Add Gaussian noise (physically accurate for FBP-reconstructed CT)."""
    return img + rng.normal(0, sigma, img.shape)


def _add_impulse_noise(img, fraction, rng):
    """Add impulse (salt-and-pepper) noise — stress test only.

    Not a real CT phenomenon: real detector failures produce ring artifacts.
    Tests the median filter's ability to handle extreme outlier voxels.
    """
    out = img.copy()
    n = int(fraction * img.size)
    coords = (rng.randint(0, img.shape[0], n), rng.randint(0, img.shape[1], n))
    out[coords] = rng.uniform(200, 1000, n)
    coords = (rng.randint(0, img.shape[0], n), rng.randint(0, img.shape[1], n))
    out[coords] = rng.uniform(-1000, -500, n)
    return out


def _add_patient_motion(img, shift_px, rotation_deg, rng, n_substeps=12):
    """Multi-step motion trajectory blending.

    Simulates motion during gantry rotation by blending sub-exposures along
    a smooth bell-shaped motion trajectory. Produces characteristic edge
    doubling and directional blur.
    """
    out = np.zeros_like(img)
    weights = np.zeros(n_substeps)

    for i in range(n_substeps):
        t = i / (n_substeps - 1)
        motion_phase = np.exp(-4 * (t - 0.5) ** 2)

        dy = shift_px * motion_phase * (0.8 + 0.2 * rng.randn())
        dx = shift_px * 0.3 * motion_phase * (0.8 + 0.2 * rng.randn())
        angle = rotation_deg * motion_phase * (0.8 + 0.2 * rng.randn())

        shifted = ndimage.shift(img, [dy, dx], order=1)
        rotated = ndimage.rotate(shifted, angle, reshape=False, order=1)

        w = 1.0
        weights[i] = w
        out += w * rotated

    return out / weights.sum()


def _add_motion_blur(img, length, axis=1):
    """Directional box-kernel motion blur (legacy helper for backward compat tests)."""
    kernel = np.zeros((length, length))
    mid = length // 2
    if axis == 1:
        kernel[mid, :] = 1.0 / length
    else:
        kernel[:, mid] = 1.0 / length
    return ndimage.convolve(img, kernel)


def _add_ring_artifacts(img, n_rings, amplitude, rng):
    """Thin discrete concentric rings from bad detector elements.

    Physically accurate: each ring is 1-2 px wide at a specific radius.
    """
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_radius = min(cy, cx) * 0.85

    out = img.copy()
    for _ in range(n_rings):
        radius = rng.uniform(20, max_radius)
        ring_width = rng.uniform(1.0, 2.0)
        ring_amp = rng.uniform(-amplitude, amplitude)
        ring = ring_amp * np.exp(-0.5 * ((dist - radius) / (ring_width * 0.5)) ** 2)
        out += ring
    return out


def _add_photon_starvation(img, amplitude, n_streaks, rng):
    """Posterior fossa streaks (Hounsfield bar) — anatomically guided.

    Places streaks at the posterior fossa level (60-70% down from top),
    between the densest lateral structures (petrous bone locations).
    """
    out = img.copy()
    h, w = img.shape
    center_y = int(h * rng.uniform(0.60, 0.70))

    # Find dense lateral structures at fossa level
    row = img[min(center_y, h - 1), :]
    mid = w // 2
    left_peak = np.argmax(row[:mid]) if mid > 0 else 0
    right_peak = mid + np.argmax(row[mid:])

    for _ in range(n_streaks):
        y_pos = center_y + rng.randint(-8, 9)
        width = rng.randint(2, 6)
        streak_amp = rng.uniform(-amplitude, amplitude)

        x_start = max(0, left_peak - 10)
        x_end = min(w, right_peak + 10)

        for row_idx in range(max(0, y_pos - width // 2),
                             min(h, y_pos + width // 2 + 1)):
            vert_w = np.exp(-0.5 * ((row_idx - y_pos) / max(1, width * 0.3)) ** 2)
            x_prof = np.zeros(w)
            x_prof[x_start:x_end] = 1.0
            taper = 8
            if x_end - x_start > 2 * taper:
                x_prof[x_start:x_start + taper] = np.linspace(0, 1, taper)
                x_prof[x_end - taper:x_end] = np.linspace(1, 0, taper)
            out[row_idx, :] += streak_amp * x_prof * vert_w

    return out


def _add_truncation(img, fov_fraction, rim_hu, rim_width):
    """FOV truncation with bright rim and inverse cupping.

    Stress test: rare in head CT, common in body CT.
    """
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    fov_radius = min(cy, cx) * fov_fraction

    out = img.copy()
    out[dist > fov_radius] = -1000.0

    rim_mask = (dist >= fov_radius - rim_width) & (dist <= fov_radius)
    out[rim_mask] += rim_hu

    inside = dist <= fov_radius
    cupping = 5.0 * (dist[inside] / fov_radius) ** 2
    out[inside] += cupping

    return out


def _add_metal_streaks(img, metal_pos, metal_hu, metal_radius,
                       streak_peak=250):
    """Metal streak artifacts via sinogram-domain Radon transform.

    Forward-projects, inserts metal, adds photon-starvation noise to the
    sinogram, reconstructs with FBP. Preserves natural near-metal → far-field
    streak falloff.
    """
    h, w = img.shape
    rng = np.random.RandomState(42)

    # Pad to diagonal so full image fits inside reconstruction circle
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2))) + 2
    size = diag if diag % 2 == 0 else diag + 1
    padded = np.zeros((size, size), dtype=np.float64)
    y0, x0 = (size - h) // 2, (size - w) // 2
    padded[y0:y0 + h, x0:x0 + w] = img + 1000.0

    yy, xx = np.ogrid[:size, :size]
    metal_mask_pad = np.zeros((size, size), dtype=bool)
    for (my, mx) in metal_pos:
        py, px = my + y0, mx + x0
        d = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
        metal_mask_pad |= (d <= metal_radius)

    img_with_metal = padded.copy()
    img_with_metal[metal_mask_pad] = metal_hu + 1000.0

    theta = np.linspace(0., 180., 360, endpoint=False)
    sino_clean = radon(padded, theta=theta, circle=True)
    sino_metal = radon(img_with_metal, theta=theta, circle=True)

    # Photon starvation noise in metal shadow
    sino_diff = sino_metal - sino_clean
    metal_shadow = sino_diff / (sino_diff.max() + 1e-10)
    starvation_noise = rng.normal(0, 1, sino_metal.shape)
    starvation_noise *= metal_shadow * sino_diff.max() * 0.08
    sino_corrupted = sino_metal + starvation_noise

    recon_corrupted = iradon(sino_corrupted, theta=theta, circle=True,
                             filter_name='ramp')
    recon_clean = iradon(sino_clean, theta=theta, circle=True,
                         filter_name='ramp')

    streak = (recon_corrupted - recon_clean)[y0:y0 + h, x0:x0 + w]

    # Zero out metal objects from streak map
    metal_mask_orig = np.zeros((h, w), dtype=bool)
    for (my, mx) in metal_pos:
        d = np.sqrt((yy[:h, :w] - my) ** 2 + (xx[:h, :w] - mx) ** 2)
        metal_mask_orig |= (d <= metal_radius + 2)
    streak[metal_mask_orig] = 0

    # Scale peak preserving natural falloff
    raw_peak = np.percentile(np.abs(streak[~metal_mask_orig]), 99.5)
    if raw_peak > 0:
        streak *= streak_peak / raw_peak

    out = img.copy()
    out += streak
    for (my, mx) in metal_pos:
        d = np.sqrt((yy[:h, :w] - my) ** 2 + (xx[:h, :w] - mx) ** 2)
        out[d <= metal_radius] = metal_hu
    return out


# ---------------------------------------------------------------------------
# Gaussian noise tests (physically accurate)
# ---------------------------------------------------------------------------

class TestGaussianNoise:
    """Verify mask quality under Gaussian noise at various sigma levels.

    Gaussian noise is physically accurate for FBP-reconstructed CT.
    sigma ~5 = standard dose, ~15 = low-dose, ~30 = ultra-low-dose.
    """

    @pytest.mark.parametrize("sigma", [5, 15, 30])
    def test_mask_still_works(self, sigma):
        rng = np.random.RandomState(42)
        img = _add_gaussian_noise(_make_phantom(), sigma, rng)
        gt = _ground_truth()
        mask = create_brain_mask(img, verbose=False)
        d = _dice(mask, gt)
        assert d > 0.5, f"Dice {d:.3f} too low at sigma={sigma}"

    @pytest.mark.parametrize("sigma", [15, 30])
    def test_median_filter_improves_quality(self, sigma):
        rng = np.random.RandomState(42)
        img = _add_gaussian_noise(_make_phantom(), sigma, rng)
        gt = _ground_truth()
        dice_plain = _dice(create_brain_mask(img, verbose=False), gt)
        dice_median = _dice(
            create_brain_mask(img, median_size=3, verbose=False), gt
        )
        assert dice_median >= dice_plain - 0.05, (
            f"Median filter hurt: {dice_median:.3f} vs {dice_plain:.3f}"
        )


# ---------------------------------------------------------------------------
# Impulse noise tests (stress test — not a real CT artifact)
# ---------------------------------------------------------------------------

class TestImpulseNoise:
    """Stress test only: salt-and-pepper is not a real CT phenomenon."""

    @pytest.mark.parametrize("fraction", [0.01, 0.05, 0.10])
    def test_median_filter_handles_impulse(self, fraction):
        rng = np.random.RandomState(42)
        img = _add_impulse_noise(_make_phantom(), fraction, rng)
        gt = _ground_truth()
        mask = create_brain_mask(img, median_size=3, verbose=False)
        d = _dice(mask, gt)
        assert d > 0.7, (
            f"Dice {d:.3f} too low with median at {fraction*100:.0f}% impulse noise"
        )


# ---------------------------------------------------------------------------
# Patient motion tests (multi-step trajectory)
# ---------------------------------------------------------------------------

class TestPatientMotion:
    """Multi-step motion trajectory with translational + rotational components."""

    @pytest.mark.parametrize("shift_px,rotation_deg", [
        (2, 1.0), (4, 2.0), (6, 3.0)
    ])
    def test_motion_with_rotation(self, shift_px, rotation_deg):
        rng = np.random.RandomState(42)
        img = _add_patient_motion(_make_phantom(), shift_px, rotation_deg, rng)
        gt = _ground_truth()
        mask = create_brain_mask(img, median_size=3, opening=True, verbose=False)
        d = _dice(mask, gt)
        assert d > 0.7, (
            f"Dice {d:.3f} too low at {shift_px}px + {rotation_deg}° motion"
        )

    @pytest.mark.parametrize("length", [3, 5, 10])
    def test_opening_helps_with_blur(self, length):
        """Legacy directional blur test — opening should not hurt."""
        img = _add_motion_blur(_make_phantom(), length)
        gt = _ground_truth()
        dice_plain = _dice(create_brain_mask(img, verbose=False), gt)
        dice_opening = _dice(
            create_brain_mask(img, opening=True, verbose=False), gt
        )
        assert dice_opening >= dice_plain - 0.05, (
            f"Opening hurt: {dice_opening:.3f} vs {dice_plain:.3f} "
            f"at blur length={length}"
        )


# ---------------------------------------------------------------------------
# Ring artifact tests (physically accurate — thin discrete rings)
# ---------------------------------------------------------------------------

class TestRingArtifacts:
    """Thin discrete rings from individual bad detector elements."""

    @pytest.mark.parametrize("n_rings,amplitude", [(1, 30), (3, 30), (5, 30)])
    def test_median_reduces_ring_impact(self, n_rings, amplitude):
        rng = np.random.RandomState(42)
        img = _add_ring_artifacts(_make_phantom(), n_rings, amplitude, rng)
        gt = _ground_truth()
        dice_plain = _dice(create_brain_mask(img, verbose=False), gt)
        dice_median = _dice(
            create_brain_mask(img, median_size=5, verbose=False), gt
        )
        assert dice_median >= dice_plain - 0.05, (
            f"Median hurt: {dice_median:.3f} vs {dice_plain:.3f} "
            f"at {n_rings} rings, amplitude={amplitude}"
        )


# ---------------------------------------------------------------------------
# Photon starvation tests (anatomically-guided approximation)
# ---------------------------------------------------------------------------

class TestPhotonStarvation:
    """Posterior fossa streaks (Hounsfield bar) — most common head CT artifact."""

    @pytest.mark.parametrize("amplitude", [20, 40, 60])
    def test_mask_still_reasonable(self, amplitude):
        rng = np.random.RandomState(42)
        img = _add_photon_starvation(_make_phantom(), amplitude, 3, rng)
        gt = _ground_truth()
        mask = create_brain_mask(img, verbose=False)
        d = _dice(mask, gt)
        assert d > 0.7, (
            f"Dice {d:.3f} too low at starvation amplitude={amplitude}"
        )

    @pytest.mark.parametrize("amplitude", [20, 40])
    def test_median_helps_with_starvation(self, amplitude):
        rng = np.random.RandomState(42)
        img = _add_photon_starvation(_make_phantom(), amplitude, 3, rng)
        gt = _ground_truth()
        dice_plain = _dice(create_brain_mask(img, verbose=False), gt)
        dice_median = _dice(
            create_brain_mask(img, median_size=3, verbose=False), gt
        )
        assert dice_median >= dice_plain - 0.05, (
            f"Median hurt with starvation: {dice_median:.3f} vs {dice_plain:.3f}"
        )


# ---------------------------------------------------------------------------
# Metal streak artifact tests (sinogram-domain via Radon transform)
# ---------------------------------------------------------------------------

class TestMetalStreaks:
    """Metal streak artifacts simulated via Radon forward/back-projection.

    Dental fillings at typical molar positions on the phantom. The
    sinogram inconsistency produces realistic bright/dark streaks.
    """

    def test_metal_streaks_mask_reasonable(self):
        """Mask should still be reasonable with dental filling streaks."""
        h, w = 256, 256
        metal_pos = [(int(h * 0.78), int(w * 0.35)),
                     (int(h * 0.78), int(w * 0.65))]
        img = _add_metal_streaks(_make_phantom(), metal_pos, 3000, 3)
        gt = _ground_truth()
        mask = create_brain_mask(img, median_size=7, verbose=False)
        d = _dice(mask, gt)
        assert d > 0.7, f"Dice {d:.3f} too low with metal streaks"

    def test_metal_streaks_default_vs_robust(self):
        """Robust params should help with metal streak artifacts."""
        h, w = 256, 256
        metal_pos = [(int(h * 0.78), int(w * 0.35)),
                     (int(h * 0.78), int(w * 0.65))]
        img = _add_metal_streaks(_make_phantom(), metal_pos, 3000, 3)
        gt = _ground_truth()
        dice_plain = _dice(create_brain_mask(img, verbose=False), gt)
        dice_robust = _dice(
            create_brain_mask(img, median_size=7, verbose=False), gt
        )
        assert dice_robust >= dice_plain - 0.05, (
            f"Robust hurt with metal: {dice_robust:.3f} vs {dice_plain:.3f}"
        )


# ---------------------------------------------------------------------------
# Truncation artifact tests (stress test — rare in head CT)
# ---------------------------------------------------------------------------

class TestTruncation:
    """FOV truncation with bright rim — stress test, rare in head CT."""

    @pytest.mark.parametrize("fov_fraction", [0.85, 0.75, 0.65])
    def test_mask_with_truncation(self, fov_fraction):
        img = _add_truncation(_make_phantom(), fov_fraction, rim_hu=30, rim_width=3)
        gt = _ground_truth()
        mask = create_brain_mask(img, median_size=3, verbose=False)
        d = _dice(mask, gt)
        assert d > 0.5, (
            f"Dice {d:.3f} too low at FOV fraction={fov_fraction}"
        )


# ---------------------------------------------------------------------------
# Combined robustness tests
# ---------------------------------------------------------------------------

class TestCombined:
    """Combined artifact scenarios — stress tests."""

    def test_combined_on_blurred_noisy(self):
        rng = np.random.RandomState(42)
        img = _make_phantom()
        img = _add_gaussian_noise(img, sigma=15, rng=rng)
        img = _add_motion_blur(img, length=5)
        gt = _ground_truth()
        mask = create_brain_mask(img, median_size=3, opening=2, verbose=False)
        d = _dice(mask, gt)
        assert d > 0.7, f"Combined Dice {d:.3f} too low"

    def test_combined_impulse_and_rings(self):
        rng = np.random.RandomState(42)
        img = _make_phantom()
        img = _add_impulse_noise(img, fraction=0.03, rng=rng)
        img = _add_ring_artifacts(img, n_rings=3, amplitude=15, rng=rng)
        gt = _ground_truth()
        mask = create_brain_mask(img, median_size=5, verbose=False)
        d = _dice(mask, gt)
        assert d > 0.7, f"Combined impulse+ring Dice {d:.3f} too low"

    def test_combined_noise_motion_cupping(self):
        rng = np.random.RandomState(42)
        img = _make_phantom()
        img = _add_gaussian_noise(img, sigma=15, rng=rng)
        img = _add_patient_motion(img, shift_px=2, rotation_deg=1.0, rng=rng)
        img += -10 * (1 - (np.sqrt(
            (np.arange(256)[:, None] - 128) ** 2 +
            (np.arange(256)[None, :] - 128) ** 2
        ) / 181.0) ** 2)
        gt = _ground_truth()
        mask = create_brain_mask(img, median_size=3, opening=2, verbose=False)
        d = _dice(mask, gt)
        assert d > 0.7, f"Combined noise+motion+cupping Dice {d:.3f} too low"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Default parameters must produce identical results to pre-0.4.0 behavior."""

    def test_defaults_identical(self):
        img = _make_phantom()
        old_mask = (img > 20) & (img < 80)
        old_mask = ndimage.binary_fill_holes(old_mask)
        labeled, n = ndimage.label(old_mask)
        if n > 1:
            sizes = ndimage.sum(old_mask, labeled, range(1, n + 1))
            largest = np.argmax(sizes) + 1
            old_mask = labeled == largest
        old_mask = ndimage.binary_fill_holes(old_mask).astype(bool)

        new_mask = create_brain_mask(img, verbose=False)
        np.testing.assert_array_equal(new_mask, old_mask)

    def test_defaults_identical_with_noise(self):
        rng = np.random.RandomState(123)
        img = _add_gaussian_noise(_make_phantom(), sigma=10, rng=rng)

        old_mask = (img > 20) & (img < 80)
        old_mask = ndimage.binary_fill_holes(old_mask)
        labeled, n = ndimage.label(old_mask)
        if n > 1:
            sizes = ndimage.sum(old_mask, labeled, range(1, n + 1))
            largest = np.argmax(sizes) + 1
            old_mask = labeled == largest
        old_mask = ndimage.binary_fill_holes(old_mask).astype(bool)

        new_mask = create_brain_mask(img, verbose=False)
        np.testing.assert_array_equal(new_mask, old_mask)

    def test_opening_iterations(self):
        img = _make_phantom()
        mask_true = create_brain_mask(img, opening=True, verbose=False)
        mask_1 = create_brain_mask(img, opening=1, verbose=False)
        mask_3 = create_brain_mask(img, opening=3, verbose=False)
        np.testing.assert_array_equal(mask_true, mask_1)
        assert mask_3.dtype == bool

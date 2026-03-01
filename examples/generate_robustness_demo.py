"""Generate robustness demonstration images for the README.

Uses real CT brain data (CT_Philips NIfTI, slice 157) with clinically
realistic artifact simulation at levels seen in actual clinical CT scanners.

Simulation methods:
- Sinogram-domain (via Radon transform): metal streak artifacts
- Physically accurate image-domain: quantum noise, beam hardening, ring artifacts
- Multi-step image-domain approximation: patient motion, photon starvation
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import radon, iradon

from ct_brain_mask import create_brain_mask

# ---------------------------------------------------------------------------
# Load real brain CT slice
# ---------------------------------------------------------------------------

NIFTI_PATH = os.path.join(os.path.dirname(__file__),
                          'isles24_sample', 'brain_ct.nii.gz')


def load_brain_slice():
    """Load CT_Philips slice 157 — the same slice used in the README demo."""
    import nibabel as nib
    data = nib.load(NIFTI_PATH).get_fdata()
    return np.rot90(data[:, :, 157]).astype(np.float64)


# ---------------------------------------------------------------------------
# Clinically realistic artifact simulation
# ---------------------------------------------------------------------------

def add_quantum_noise(img, sigma, rng):
    """Quantum / electronic noise (Gaussian).

    Physically accurate: FBP-reconstructed CT images have approximately
    Gaussian noise. sigma ~5 HU = standard dose, ~15 HU = low-dose,
    ~30 HU = ultra-low-dose.
    """
    return img + rng.normal(0, sigma, img.shape)


def add_patient_motion(img, shift_px, rotation_deg, rng, n_substeps=12):
    """Patient head motion — multi-step trajectory blending.

    Simulates motion during gantry rotation by blending many sub-exposures
    along a smooth motion trajectory. Each sub-step applies a fraction of
    the total translational and rotational displacement, weighted by the
    fraction of gantry angles acquired during that motion phase.

    More realistic than simple 2-copy blending: produces the characteristic
    edge doubling and directional blur seen in clinical motion artifacts.
    """
    out = np.zeros_like(img)
    weights = np.zeros(n_substeps)

    for i in range(n_substeps):
        # Motion ramps up mid-scan (patient startles, coughs, etc.)
        t = i / (n_substeps - 1)  # 0 to 1
        # Bell-shaped motion profile: most displacement mid-scan
        motion_phase = np.exp(-4 * (t - 0.5) ** 2)

        dy = shift_px * motion_phase * (0.8 + 0.2 * rng.randn())
        dx = shift_px * 0.3 * motion_phase * (0.8 + 0.2 * rng.randn())
        angle = rotation_deg * motion_phase * (0.8 + 0.2 * rng.randn())

        shifted = ndimage.shift(img, [dy, dx], order=1)
        rotated = ndimage.rotate(shifted, angle, reshape=False, order=1)

        # Weight: uniform across gantry angles (each sub-step = equal arc)
        w = 1.0
        weights[i] = w
        out += w * rotated

    return out / weights.sum()


def add_ring_artifacts(img, n_rings, amplitude, rng):
    """Ring artifacts from miscalibrated/defective detector elements.

    Physically accurate model: each bad detector element produces a thin
    (1-2 px wide) concentric ring at a specific radius from the isocenter.
    Amplitude +-10-60 HU per ring. Rings are always centered on image center.
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


def add_beam_hardening(img, strength):
    """Beam hardening — cupping artifact.

    Physically accurate: lower-energy photons are preferentially absorbed
    by dense bone, making the center of the brain appear darker. The
    parabolic profile is the standard model used in correction algorithms.
    Strength 5-15 HU = residual after correction, 20-40 HU = uncorrected.
    """
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_dist = np.sqrt(cy**2 + cx**2)
    cupping = -strength * (1 - (dist / max_dist) ** 2)
    return img + cupping


def add_photon_starvation(img, amplitude, rng):
    """Photon starvation streaks between petrous temporal bones.

    Simulates the Hounsfield bar: the single most common artifact in routine
    head CT. Dense petrous bones cause photon starvation at specific gantry
    angles, producing dark/bright streaks connecting the two petrous bones
    through the posterior fossa.

    Uses the actual bone locations in the image to place streaks along the
    line connecting the densest lateral structures at posterior fossa level.
    """
    out = img.copy()
    h, w = img.shape

    # Find posterior fossa level (60-70% down from top)
    fossa_y = int(h * 0.65)

    # Find the two densest lateral points at fossa level (petrous bones)
    row = img[fossa_y, :]
    mid = w // 2
    left_peak = np.argmax(row[:mid])
    right_peak = mid + np.argmax(row[mid:])

    # Create streak pattern between petrous bones
    for _ in range(3):
        y_pos = fossa_y + rng.randint(-8, 9)
        streak_width = rng.randint(2, 6)
        streak_amp = rng.uniform(-amplitude, amplitude)

        # Streak extends between the two petrous bones with some overshoot
        x_start = max(0, left_peak - 10)
        x_end = min(w, right_peak + 10)

        for row_idx in range(max(0, y_pos - streak_width // 2),
                             min(h, y_pos + streak_width // 2 + 1)):
            vert_w = np.exp(-0.5 * ((row_idx - y_pos) / max(1, streak_width * 0.3)) ** 2)
            # Horizontal profile: strong between bones, tapering outside
            x_prof = np.zeros(w)
            x_prof[x_start:x_end] = 1.0
            # Smooth taper at edges
            taper = 8
            if x_end - x_start > 2 * taper:
                x_prof[x_start:x_start + taper] = np.linspace(0, 1, taper)
                x_prof[x_end - taper:x_end] = np.linspace(1, 0, taper)
            out[row_idx, :] += streak_amp * x_prof * vert_w

    return out


def add_metal_streaks(img, metal_pos, metal_hu, metal_radius, rng,
                      streak_peak=250):
    """Metal streak artifacts via sinogram-domain simulation.

    Forward-projects the image with metal inserted, adds photon-starvation
    noise to sinogram lines passing through the metal, then reconstructs
    with FBP. The sinogram inconsistency + starvation noise produces the
    characteristic bright/dark streaks radiating from metal objects.

    Parameters
    ----------
    metal_pos : list of (row, col) tuples — metal object center positions
    metal_hu : float — metal attenuation in HU (dental amalgam ~3000)
    metal_radius : int — radius of each metal object in pixels
    streak_peak : float — desired peak streak amplitude in HU near the metal
        (clinical range: 100-500 HU for dental fillings)
    """
    h, w = img.shape

    # Pad to diagonal so the full image fits inside the reconstruction
    # circle — eliminates the "must be zero outside" warning
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2))) + 2
    size = diag if diag % 2 == 0 else diag + 1
    padded = np.zeros((size, size), dtype=np.float64)
    y0, x0 = (size - h) // 2, (size - w) // 2
    # Shift to non-negative for Radon (air at 0 instead of -1000)
    padded[y0:y0 + h, x0:x0 + w] = img + 1000.0

    # Metal mask on padded grid
    yy, xx = np.ogrid[:size, :size]
    metal_mask_pad = np.zeros((size, size), dtype=bool)
    for (my, mx) in metal_pos:
        py, px = my + y0, mx + x0
        d = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
        metal_mask_pad |= (d <= metal_radius)

    img_with_metal = padded.copy()
    img_with_metal[metal_mask_pad] = metal_hu + 1000.0

    # Forward project — 360 angles for good angular sampling
    theta = np.linspace(0., 180., 360, endpoint=False)
    sino_clean = radon(padded, theta=theta, circle=True)
    sino_metal = radon(img_with_metal, theta=theta, circle=True)

    # Photon starvation: projections through metal have very few photons,
    # so they're much noisier. Add noise proportional to metal attenuation
    # in each sinogram bin.
    sino_diff = sino_metal - sino_clean
    metal_shadow = sino_diff / (sino_diff.max() + 1e-10)  # 0-1 normalized
    starvation_noise = rng.normal(0, 1, sino_metal.shape)
    # Scale noise: strongest where metal shadow is strongest
    starvation_noise *= metal_shadow * sino_diff.max() * 0.08
    sino_corrupted = sino_metal + starvation_noise

    # Reconstruct
    recon_corrupted = iradon(sino_corrupted, theta=theta, circle=True,
                             filter_name='ramp')
    recon_clean = iradon(sino_clean, theta=theta, circle=True,
                         filter_name='ramp')

    # Extract streak pattern (difference = metal + streaks + starvation)
    streak = (recon_corrupted - recon_clean)[y0:y0 + h, x0:x0 + w]

    # Zero out the metal object itself from the streak map
    metal_mask_orig = np.zeros((h, w), dtype=bool)
    for (my, mx) in metal_pos:
        d = np.sqrt((yy[:h, :w] - my) ** 2 + (xx[:h, :w] - mx) ** 2)
        metal_mask_orig |= (d <= metal_radius + 2)
    streak[metal_mask_orig] = 0

    # Scale so peak streak amplitude matches desired clinical level,
    # preserving the natural near-metal → far-field falloff
    raw_peak = np.percentile(np.abs(streak[~metal_mask_orig]), 99.5)
    if raw_peak > 0:
        streak *= streak_peak / raw_peak

    out = img.copy()
    out += streak
    # Paint metal objects at metal HU
    for (my, mx) in metal_pos:
        d = np.sqrt((yy[:h, :w] - my) ** 2 + (xx[:h, :w] - mx) ** 2)
        out[d <= metal_radius] = metal_hu
    return out


# ---------------------------------------------------------------------------
# Figure 1: Robust mask under clinical CT artifacts
# ---------------------------------------------------------------------------

def generate_artifact_comparison():
    brain = load_brain_slice()
    gt = create_brain_mask(brain, verbose=False)
    h, w = brain.shape

    metal_positions = [
        (int(h * 0.78), int(w * 0.35)),
        (int(h * 0.78), int(w * 0.65)),
    ]

    scenarios = [
        ("Clean\n(reference)",
         brain, {}),
        ("Low-dose Noise\n(σ=15 HU)",
         add_quantum_noise(brain, 15, np.random.RandomState(42)),
         {"median_size": 3}),
        ("Patient Motion\n(3px + 1°)",
         add_patient_motion(brain, 3, 1.0, np.random.RandomState(42)),
         {"median_size": 3, "opening": True}),
        ("Ring Artifacts\n(3 rings, 30 HU)",
         add_ring_artifacts(brain, 3, 30, np.random.RandomState(42)),
         {"median_size": 5}),
        ("Beam Hardening\n(40 HU cupping)",
         add_beam_hardening(brain, 40),
         {"median_size": 3}),
        ("Photon Starvation\n(posterior fossa)",
         add_photon_starvation(brain, 40, np.random.RandomState(42)),
         {"median_size": 3}),
        ("Metal Streaks\n(dental fillings)",
         add_metal_streaks(brain, metal_positions, 3000, 3,
                           np.random.RandomState(42)),
         {"median_size": 7}),
    ]

    n = len(scenarios)
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8.0),
                             gridspec_kw={"hspace": 0.10, "wspace": 0.05})

    row_labels = ["Degraded CT", "Robust mask", "Overlay"]

    for col, (label, img, robust_kw) in enumerate(scenarios):
        mask_robust = create_brain_mask(img, verbose=False, **robust_kw) if robust_kw else create_brain_mask(img, verbose=False)
        d_robust = dice(mask_robust, gt)

        # Row 0: degraded CT (brain window)
        ax = axes[0, col]
        ax.imshow(np.clip(img, -10, 100), cmap="gray", interpolation="none")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.axis("off")

        # Row 1: robust mask
        ax = axes[1, col]
        ax.imshow(mask_robust, cmap="gray", interpolation="none")
        if robust_kw:
            param_str = ", ".join(f"{k}={v}" for k, v in robust_kw.items())
            ax.set_xlabel(f"Dice = {d_robust:.3f}\n({param_str})", fontsize=7,
                          color="#4E8C2F")
        else:
            ax.set_xlabel(f"Dice = {d_robust:.3f}\n(default)", fontsize=7,
                          color="#4E8C2F")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Row 2: overlay on brain CT
        ax = axes[2, col]
        bg = np.clip((img - (-10)) / (100 - (-10)), 0, 1)
        overlay = np.stack([bg, bg, bg], axis=-1).astype(np.float32)
        tp = mask_robust & gt
        fp = mask_robust & ~gt
        fn = ~mask_robust & gt
        overlay[tp] = [0.15, 0.7, 0.15]
        overlay[fp] = [0.9, 0.15, 0.15]
        overlay[fn] = [0.2, 0.35, 0.9]
        ax.imshow(overlay, interpolation="none")
        ax.axis("off")

    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=9, fontweight="bold", rotation=0,
                                labelpad=65, va="center")

    fig.suptitle("Robustness to Clinical CT Artifacts — Real Brain (CT_Philips, slice 157)",
                 fontsize=12, fontweight="bold", y=0.98)

    fig.savefig("examples/robustness_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved examples/robustness_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: Dice bar chart — robust mask quality
# ---------------------------------------------------------------------------

def generate_dice_chart():
    brain = load_brain_slice()
    gt = create_brain_mask(brain, verbose=False)
    h, w = brain.shape

    conditions = []
    dice_values = []
    robust_params = []

    # Quantum noise (physically accurate)
    for sigma in [5, 15, 30]:
        img = add_quantum_noise(brain, sigma, np.random.RandomState(42))
        d = dice(create_brain_mask(img, median_size=3, verbose=False), gt)
        conditions.append(f"Noise\nσ={sigma}")
        dice_values.append(d)
        robust_params.append("median_size=3")

    # Patient motion (multi-step approximation)
    for shift in [2, 4, 6]:
        rot = shift * 0.5
        img = add_patient_motion(brain, shift, rot, np.random.RandomState(42))
        d = dice(create_brain_mask(img, median_size=3, opening=True, verbose=False), gt)
        conditions.append(f"Motion\n{shift}px+{rot:.0f}°")
        dice_values.append(d)
        robust_params.append("median_size=3, opening=True")

    # Ring artifacts (physically accurate — thin discrete rings)
    for n_rings in [1, 3, 5]:
        img = add_ring_artifacts(brain, n_rings, 30, np.random.RandomState(42))
        d = dice(create_brain_mask(img, median_size=5, verbose=False), gt)
        conditions.append(f"Rings\n{n_rings}×30HU")
        dice_values.append(d)
        robust_params.append("median_size=5")

    # Beam hardening / cupping (physically accurate)
    for strength in [10, 20, 40]:
        img = add_beam_hardening(brain, strength)
        d = dice(create_brain_mask(img, median_size=3, verbose=False), gt)
        conditions.append(f"Cupping\n{strength} HU")
        dice_values.append(d)
        robust_params.append("median_size=3")

    # Photon starvation (anatomically-guided approximation)
    for amp in [20, 40]:
        img = add_photon_starvation(brain, amp, np.random.RandomState(42))
        d = dice(create_brain_mask(img, median_size=3, verbose=False), gt)
        conditions.append(f"Starvation\n{amp} HU")
        dice_values.append(d)
        robust_params.append("median_size=3")

    # Metal streak artifacts (sinogram-domain via Radon transform)
    metal_pos = [(int(h * 0.78), int(w * 0.35)),
                 (int(h * 0.78), int(w * 0.65))]
    img = add_metal_streaks(brain, metal_pos, 3000, 3, np.random.RandomState(42))
    d = dice(create_brain_mask(img, median_size=7, verbose=False), gt)
    conditions.append("Metal\nStreaks")
    dice_values.append(d)
    robust_params.append("median_size=7")

    # Combined (stress test: noise + motion + cupping)
    img = add_beam_hardening(
        add_patient_motion(
            add_quantum_noise(brain, 15, np.random.RandomState(99)),
            2, 0.5, np.random.RandomState(42)),
        10)
    d = dice(create_brain_mask(img, median_size=3, opening=2, verbose=False), gt)
    conditions.append("Combined")
    dice_values.append(d)
    robust_params.append("median_size=3, opening=2")

    # Plot
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(14, 4.5))
    bars = ax.bar(x, dice_values, 0.55, color="#70AD47", edgecolor="white")

    ax.set_ylabel("Dice Coefficient", fontsize=11)
    ax.set_title("Robust Mask Quality on Real Brain CT under Clinical Artifacts",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    y_min = min(dice_values)
    ax.set_ylim(max(0, y_min - 0.05), 1.05)
    ax.axhline(y=0.95, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    for b in bars:
        h_val = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h_val + 0.005, f"{h_val:.3f}",
                ha="center", va="bottom", fontsize=7, color="#4E8C2F",
                fontweight="bold")

    fig.savefig("examples/robustness_dice_chart.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved examples/robustness_dice_chart.png")

    # Print markdown table
    print("\n| Condition | Dice | Robust params | Simulation |")
    print("|-----------|------|---------------|------------|")
    sim_labels = (
        ["Physically accurate"] * 3 +
        ["Multi-step trajectory"] * 3 +
        ["Physically accurate"] * 3 +
        ["Physically accurate"] * 3 +
        ["Anatomy-guided"] * 2 +
        ["Sinogram-domain (Radon)"] +
        ["Stress test"]
    )
    for cond, dv, rp, sl in zip(conditions, dice_values, robust_params, sim_labels):
        cond_flat = cond.replace("\n", " ")
        print(f"| {cond_flat} | {dv:.3f} | `{rp}` | {sl} |")


# ---------------------------------------------------------------------------

def dice(mask, gt):
    intersection = np.logical_and(mask, gt).sum()
    total = mask.sum() + gt.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


if __name__ == "__main__":
    if not os.path.isfile(NIFTI_PATH):
        print(f"Error: NIfTI file not found at {NIFTI_PATH}", file=sys.stderr)
        print("Download the sample data first.", file=sys.stderr)
        sys.exit(1)
    generate_artifact_comparison()
    generate_dice_chart()
    print("\nDone!")

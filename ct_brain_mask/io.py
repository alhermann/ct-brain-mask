"""
CT Brain Mask IO — convenience loaders for DICOM and NIfTI CT data.

All medical-imaging dependencies (pydicom, nibabel) are optional.
A clear ImportError is raised if they are missing.
"""

import os
from collections import defaultdict

import numpy as np


def _require_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM loading. "
            "Install it with: pip install ct-brain-mask[io]"
        )


def _require_nibabel():
    try:
        import nibabel
        return nibabel
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI loading. "
            "Install it with: pip install ct-brain-mask[io]"
        )


def _to_hu(ds):
    """Convert a pydicom Dataset to a Hounsfield Unit numpy array."""
    pixel_array = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    return pixel_array * slope + intercept


def load_dicom_file(filepath):
    """
    Load a single DICOM file and convert to Hounsfield Units.

    Parameters
    ----------
    filepath : str or os.PathLike
        Path to a DICOM file.

    Returns
    -------
    image : np.ndarray, shape (H, W), dtype float64
        CT image in Hounsfield Units.

    Raises
    ------
    ImportError
        If pydicom is not installed.
    FileNotFoundError
        If the file does not exist.
    """
    pydicom = _require_pydicom()
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"DICOM file not found: {filepath}")
    ds = pydicom.dcmread(filepath)
    return _to_hu(ds)


def load_dicom_dir(dicom_dir):
    """
    Load all DICOM files in a directory, sort by SliceLocation, and convert to HU.

    Structural CT (one frame per slice location) returns shape ``(S, H, W)``.
    Dynamic CT (multiple frames per slice location) returns shape ``(S, H, W, T)``,
    with frames sorted by InstanceNumber within each slice location.

    Parameters
    ----------
    dicom_dir : str or os.PathLike
        Path to a directory containing DICOM files.

    Returns
    -------
    volume : np.ndarray
        ``(S, H, W)`` for structural CT or ``(S, H, W, T)`` for dynamic CT.
        Values are in Hounsfield Units (float64).

    Raises
    ------
    ImportError
        If pydicom is not installed.
    FileNotFoundError
        If the directory does not exist or contains no DICOM files.
    """
    pydicom = _require_pydicom()
    dicom_dir = str(dicom_dir)
    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    # Read all DICOM files
    datasets = []
    for fname in sorted(os.listdir(dicom_dir)):
        fpath = os.path.join(dicom_dir, fname)
        if not os.path.isfile(fpath) or fname.startswith('.'):
            continue
        try:
            ds = pydicom.dcmread(fpath)
            _ = ds.pixel_array  # verify it has pixel data
            datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise FileNotFoundError(f"No valid DICOM files found in: {dicom_dir}")

    # Group by SliceLocation
    by_location = defaultdict(list)
    for ds in datasets:
        loc = float(getattr(ds, 'SliceLocation', 0))
        by_location[loc].append(ds)

    # Sort slice locations
    sorted_locations = sorted(by_location.keys())

    # Sort frames within each location by InstanceNumber
    for loc in sorted_locations:
        by_location[loc].sort(
            key=lambda d: int(getattr(d, 'InstanceNumber', 0))
        )

    frames_per_location = max(len(by_location[loc]) for loc in sorted_locations)

    if frames_per_location == 1:
        # Structural CT: (S, H, W)
        slices = []
        for loc in sorted_locations:
            slices.append(_to_hu(by_location[loc][0]))
        return np.stack(slices, axis=0)
    else:
        # Dynamic CT: (S, H, W, T)
        volume = []
        for loc in sorted_locations:
            frames = [_to_hu(ds) for ds in by_location[loc]]
            volume.append(np.stack(frames, axis=-1))
        return np.stack(volume, axis=0)


def load_nifti(filepath):
    """
    Load a NIfTI file (.nii or .nii.gz) as a numpy array.

    Parameters
    ----------
    filepath : str or os.PathLike
        Path to a NIfTI file.

    Returns
    -------
    data : np.ndarray
        Image data from the NIfTI file.

    Raises
    ------
    ImportError
        If nibabel is not installed.
    FileNotFoundError
        If the file does not exist.
    """
    nib = _require_nibabel()
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")
    return nib.load(filepath).get_fdata()

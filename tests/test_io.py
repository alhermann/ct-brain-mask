"""Tests for ct_brain_mask.io module."""

import os
import sys
from unittest import mock

import numpy as np
import pytest

from ct_brain_mask import io as ct_io


# ---------------------------------------------------------------------------
# Paths to real data (tests skip gracefully if unavailable)
# ---------------------------------------------------------------------------

DICOM_DIR = "/Users/hermann/Desktop/SPPINNs/DICOM/0000413A"
NIFTI_CT_PATH = os.path.join(os.path.dirname(__file__),
                              '..', 'examples', 'isles24_sample', 'brain_ct.nii.gz')


# ---------------------------------------------------------------------------
# ImportError tests (work regardless of installed packages)
# ---------------------------------------------------------------------------

class TestImportErrors:
    """Verify clear ImportError messages when optional deps are missing."""

    def test_load_dicom_file_without_pydicom(self):
        """load_dicom_file should raise ImportError if pydicom missing."""
        with mock.patch.dict(sys.modules, {'pydicom': None}):
            with pytest.raises(ImportError, match="pydicom"):
                ct_io.load_dicom_file("/some/fake/path.dcm")

    def test_load_dicom_dir_without_pydicom(self):
        """load_dicom_dir should raise ImportError if pydicom missing."""
        with mock.patch.dict(sys.modules, {'pydicom': None}):
            with pytest.raises(ImportError, match="pydicom"):
                ct_io.load_dicom_dir("/some/fake/dir")

    def test_load_nifti_without_nibabel(self):
        """load_nifti should raise ImportError if nibabel missing."""
        with mock.patch.dict(sys.modules, {'nibabel': None}):
            with pytest.raises(ImportError, match="nibabel"):
                ct_io.load_nifti("/some/fake/path.nii.gz")


# ---------------------------------------------------------------------------
# FileNotFoundError tests
# ---------------------------------------------------------------------------

class TestFileNotFound:
    """Verify FileNotFoundError for bad paths."""

    def test_load_dicom_file_bad_path(self):
        with pytest.raises(FileNotFoundError, match="DICOM file not found"):
            ct_io.load_dicom_file("/nonexistent/path/to/file.dcm")

    def test_load_dicom_dir_bad_path(self):
        with pytest.raises(FileNotFoundError, match="DICOM directory not found"):
            ct_io.load_dicom_dir("/nonexistent/path/to/dir")

    def test_load_nifti_bad_path(self):
        with pytest.raises(FileNotFoundError, match="NIfTI file not found"):
            ct_io.load_nifti("/nonexistent/path/to/file.nii.gz")


# ---------------------------------------------------------------------------
# Tests with real DICOM data (skipped if unavailable)
# ---------------------------------------------------------------------------

class TestLoadDicomFile:
    """Tests for load_dicom_file with real DICOM data."""

    def _get_first_dicom(self):
        if not os.path.isdir(DICOM_DIR):
            pytest.skip("DICOM data not available")
        files = sorted(f for f in os.listdir(DICOM_DIR) if not f.startswith('.'))
        if not files:
            pytest.skip("No DICOM files found")
        return os.path.join(DICOM_DIR, files[0])

    def test_loads_2d_image(self):
        """load_dicom_file should return a 2D HU array."""
        fpath = self._get_first_dicom()
        img = ct_io.load_dicom_file(fpath)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2
        assert img.dtype == np.float64

    def test_hu_range_reasonable(self):
        """Loaded DICOM should have typical CT HU range."""
        fpath = self._get_first_dicom()
        img = ct_io.load_dicom_file(fpath)
        assert img.min() < 0, "CT should have air (negative HU)"
        assert img.max() > 80, "CT should have bone (> 80 HU)"


class TestLoadDicomDir:
    """Tests for load_dicom_dir with real DICOM data."""

    def test_loads_volume(self):
        """load_dicom_dir should return a 3D or 4D array."""
        if not os.path.isdir(DICOM_DIR):
            pytest.skip("DICOM data not available")
        vol = ct_io.load_dicom_dir(DICOM_DIR)
        assert isinstance(vol, np.ndarray)
        assert vol.ndim in (3, 4)
        assert vol.dtype == np.float64

    def test_empty_dir(self, tmp_path):
        """Empty directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No valid DICOM"):
            ct_io.load_dicom_dir(tmp_path)


# ---------------------------------------------------------------------------
# Tests with NIfTI data (skipped if unavailable)
# ---------------------------------------------------------------------------

class TestLoadNifti:
    """Tests for load_nifti with real NIfTI data."""

    def test_loads_3d_volume(self):
        """load_nifti should return a 3D array from the sample CT."""
        if not os.path.isfile(NIFTI_CT_PATH):
            pytest.skip("NIfTI CT data not available")
        data = ct_io.load_nifti(NIFTI_CT_PATH)
        assert isinstance(data, np.ndarray)
        assert data.ndim == 3

    def test_hu_range_reasonable(self):
        """Loaded NIfTI should have typical CT HU range."""
        if not os.path.isfile(NIFTI_CT_PATH):
            pytest.skip("NIfTI CT data not available")
        data = ct_io.load_nifti(NIFTI_CT_PATH)
        assert data.min() < -500, "CT should have air values"
        assert data.max() > 500, "CT should have bone values"

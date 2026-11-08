"""Regression checks for the FID driver cascade; no model or data download."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fid_robust


class RobustFIDTests(unittest.TestCase):
    def test_scipy_recovers_after_numpy_svd_failure(self):
        # An independent QR driver must preserve a small singular direction.
        rng = np.random.default_rng(19)
        left, _ = np.linalg.qr(rng.normal(size=(7, 3)))
        right, _ = np.linalg.qr(rng.normal(size=(5, 3)))
        expected = np.array([5.0, 1.0, 1e-9])
        matrix = (left * expected) @ right.T
        with patch.object(np.linalg, "svd", side_effect=np.linalg.LinAlgError):
            values, path = fid_robust._svdvals(matrix)
        self.assertEqual(path, "gesvd")
        np.testing.assert_allclose(values[:3], expected, rtol=1e-6, atol=1e-14)
        np.testing.assert_allclose(values[3:], 0, atol=1e-14)

    def test_final_fallback_after_both_svd_drivers_fail(self):
        matrix = np.diag([3.0, 2.0, 1.0])
        with patch.object(np.linalg, "svd", side_effect=np.linalg.LinAlgError), \
                patch.object(scipy.linalg, "svd", side_effect=np.linalg.LinAlgError):
            values, path = fid_robust._svdvals(matrix)
        self.assertEqual(path, "eigh(A^T A)")
        np.testing.assert_allclose(np.sort(values), [1.0, 2.0, 3.0])

    def test_translated_features_have_known_fid(self):
        # Also exercises rank-deficient inputs with fewer samples than features.
        features = np.random.default_rng(23).normal(size=(5, 9))
        shift = np.arange(9, dtype=float) / 10
        actual = fid_robust.fid_from_features(features, features + shift)
        self.assertAlmostEqual(actual, float(shift @ shift), places=11)


if __name__ == "__main__":
    unittest.main()

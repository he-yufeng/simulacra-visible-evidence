"""Two transport checks, not another model or survey-accuracy test."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compact_ablation_vectors import pack, unpack, VARIANTS


class CompactTests(unittest.TestCase):
    def test_float64_round_trip_is_bit_exact_with_heterogeneous_widths(self):
        vectors = [[np.nextafter(.5, 1), .5], [.1, .2, .7], [1e-250, 1.0]]
        outputs = {variant: vectors for variant in VARIANTS}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vectors.npz"
            pack(path, outputs)
            for variant in VARIANTS:
                actual = unpack(path, variant)
                for before, after in zip(vectors, actual):
                    self.assertEqual(np.asarray(before, dtype=np.float64).tobytes(), after.tobytes())
            with self.assertRaises(FileExistsError):
                pack(path, outputs)

    def test_variant_count_and_width_incompatibility_fail_before_write(self):
        outputs = {variant: [[.5, .5]] for variant in VARIANTS}
        outputs["r09"] = [[1.0]]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vectors.npz"
            with self.assertRaises(ValueError):
                pack(path, outputs)
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                unpack(path, "unknown")


if __name__ == "__main__":
    unittest.main()

import unittest

import SimpleITK as sitk

from nnunetv2.dataset_conversion.Dataset002_cSDH import _same_geometry
from nnunetv2.dataset_conversion.Dataset004_cSDH_Abstract import split_patients


class TestCSDHConversion(unittest.TestCase):
    def test_patient_split_finds_exact_split_when_valid(self):
        sizes = [3, 2, 3, 3, 4, 3, 4, 4, 4, 3, 2]
        groups = {f"patient_{i}": [None] * size for i, size in enumerate(sizes)}

        train, test = split_patients(groups, 20, seed=42)

        self.assertEqual(sum(len(groups[name]) for name in train), 20)
        self.assertEqual(sum(len(groups[name]) for name in test), 15)
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(groups))

    def test_patient_split_rejects_impossible_target(self):
        groups = {"patient_a": [None, None], "patient_b": [None, None]}

        with self.assertRaises(ValueError):
            split_patients(groups, 1)

    def test_dicom_geometry_matches_all_physical_fields(self):
        reference = sitk.Image([8, 8, 8], sitk.sitkUInt8)
        candidate = sitk.Image([8, 8, 8], sitk.sitkUInt8)
        reference.SetSpacing((0.7, 0.7, 1.0))
        candidate.SetSpacing((0.7, 0.7, 1.0))
        reference.SetOrigin((1.0, 2.0, 3.0))
        candidate.SetOrigin((1.0, 2.0, 3.0))

        self.assertTrue(_same_geometry(candidate, reference))

        candidate.SetOrigin((1.0, 2.0, 3.1))
        self.assertFalse(_same_geometry(candidate, reference))

    def test_dicom_geometry_rejects_spacing_and_direction_mismatches(self):
        reference = sitk.Image([8, 8, 8], sitk.sitkUInt8)
        candidate = sitk.Image([8, 8, 8], sitk.sitkUInt8)

        candidate.SetSpacing((1.0, 1.0, 1.1))
        self.assertFalse(_same_geometry(candidate, reference))

        candidate.SetSpacing(reference.GetSpacing())
        candidate.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        self.assertFalse(_same_geometry(candidate, reference))


if __name__ == "__main__":
    unittest.main()

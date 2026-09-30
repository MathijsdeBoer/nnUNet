import unittest

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from nnunetv2.training.loss.cbdice_loss import _distance_transform_per_sample


class TestCBDiceLoss(unittest.TestCase):
    def test_distance_transform_is_independent_per_batch_item(self):
        mask = torch.zeros((2, 7, 7), dtype=torch.int32)
        mask[0, 2:5, 2:5] = 1
        mask[1, 1:6, 1:6] = 1

        actual = _distance_transform_per_sample(mask).numpy()
        expected = np.stack([distance_transform_edt(sample.numpy()) for sample in mask])

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()

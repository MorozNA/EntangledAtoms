import unittest
import numpy as np
from src.orbitals import PrimitiveGaussian


def gaussian_3d(coeff, alpha, R, r):
    return coeff * np.exp(-alpha * np.dot(r - R, r - R))


class TestPrimitiveGaussian(unittest.TestCase):

    def test_multiply1(self):
        alpha1, alpha2 = 0.1, 0.5
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        self.assertEqual(pg3.alpha, alpha1 + alpha2)
        coord_i = (alpha1 * R1 + alpha2 * R2) / (alpha1 + alpha2)
        for i in range(3):
            self.assertEqual(pg3.coordinates[i], coord_i[i])

    def test_multiply2(self):
        alpha1, alpha2 = 0.1, 0.5
        c1, c2 = 0.35, 0.95
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        r = np.array([0.5, 0.5, 0.2])
        value1 = gaussian_3d(c1, alpha1, R1, r)
        value2 = gaussian_3d(c2, alpha2, R2, r)
        self.assertEqual(pg1.get_value(r) * pg2.get_value(r), value1 * value2)
        round_const = 15
        self.assertEqual(np.round(pg3.get_value(r), round_const), np.round(value1 * value2, round_const))


if __name__ == '__main__':
    unittest.main()

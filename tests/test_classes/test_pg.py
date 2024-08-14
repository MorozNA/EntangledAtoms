import unittest
import numpy as np
from src.orbitals import PrimitiveGaussian


def gaussian(coeff, alpha, R, r):
    return coeff * np.exp(-np.dot(alpha, (r - R) ** 2))


class TestPrimitiveGaussian(unittest.TestCase):

    def test_multiply1(self):
        alpha1, alpha2 = np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2

        coord_i = (alpha1 * R1 + alpha2 * R2) / (alpha1 + alpha2)
        for i in range(3):
            self.assertEqual(pg3.alpha[i], alpha1[i] + alpha2[i])
            self.assertEqual(pg3.coordinates[i], coord_i[i])

    def test_multiply2(self):
        alpha1, alpha2 = np.array([0.1, 0.2, 0.3]), np.array([0.6, 0.5, 0.4])
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2

        r = np.array([0.5, 0.5, 0.2])
        value1 = gaussian(c1, alpha1, R1, r)
        value2 = gaussian(c2, alpha2, R2, r)

        round_const = 12
        self.assertEqual(np.round(pg1.get_value(r) * pg2.get_value(r), round_const),
                         np.round(value1 * value2, round_const))
        self.assertEqual(np.round(pg3.get_value(r), round_const), np.round(value1 * value2, round_const))

    def test_2D(self):
        alpha1 = np.array([0.1, 0.5])
        c1 = 0.35
        R1 = np.array([1.1, 1.2])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)

        round_const = 12
        r = np.array([0.5, 0.2])
        value1 = gaussian(c1, alpha1, R1, r)
        self.assertEqual(np.round(pg1.get_value(r), round_const), np.round(value1, round_const))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from src.orbitals import PrimitiveGaussian, GaussianProduct


def gaussian(coeff, alpha, R, r):
    return coeff * np.exp(-np.dot(alpha, (r - R) ** 2))


class TestGaussianProduct(unittest.TestCase):

    def test_get_value(self):
        alpha1, alpha2 = np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        ao1 = GaussianProduct([pg1, pg2, pg3], [0, 1, 2])

        r1, r2, r3 = np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])
        value1 = gaussian(c1, alpha1, R1, r1)
        value2 = gaussian(c2, alpha2, R2, r2)
        value3 = gaussian(pg3.coeff, pg3.alpha, pg3.coordinates, r3)
        self.assertEqual(ao1.get_value([r1, r2, r3]), value1 * value2 * value3)

    def test_multiplication1(self):
        alpha1, alpha2 = np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        ao1 = GaussianProduct([pg1, pg2, pg3], [0, 1, 2])
        ao2 = GaussianProduct([pg1, pg2, pg3], [1, 0, 2])

        ao3 = ao1 * ao2

        r1, r2, r3 = np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])

        value11 = gaussian(c1, alpha1, R1, r1)
        value12 = gaussian(c2, alpha2, R2, r2)
        value13 = gaussian(pg3.coeff, pg3.alpha, pg3.coordinates, r3)
        value21 = gaussian(c1, alpha1, R1, r2)
        value22 = gaussian(c2, alpha2, R2, r1)
        value23 = value13
        v_new_1 = value11 * value22
        v_new_2 = value12 * value21
        v_new_3 = value13 * value23

        actual_value = np.round(ao3.get_value([r1, r2, r3]), 15)
        expected_value = np.round(v_new_1 * v_new_2 * v_new_3, 15)

        self.assertEqual(actual_value, expected_value)

    def test_multiplacation2(self):
        alpha1, alpha2 = np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        ao1 = GaussianProduct([pg1], [0])
        ao2 = GaussianProduct([pg2], [1])
        ao3 = GaussianProduct([pg3], [2])

        ao_prod = ao1 * ao2 * ao3

        r1, r2, r3 = np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])

        value1 = gaussian(c1, alpha1, R1, r1)
        value2 = gaussian(c2, alpha2, R2, r2)
        value3 = gaussian(pg3.coeff, pg3.alpha, pg3.coordinates, r3)
        v_prod = value1 * value2 * value3

        self.assertEqual(ao_prod.get_value([r1, r2, r3]), v_prod)


if __name__ == '__main__':
    unittest.main()

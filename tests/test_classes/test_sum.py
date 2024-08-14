import unittest
import numpy as np
from src.orbitals import PrimitiveGaussian, GaussianProduct, Orbital


def gaussian(coeff, alpha, R, r):
    return coeff * np.exp(-np.dot(alpha, (r - R) ** 2))


class TestOrbital(unittest.TestCase):

    def test_addition(self):
        alpha1, alpha2 = np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        ao1 = GaussianProduct([pg1, pg2, pg3], [0, 1, 2])
        ao2 = GaussianProduct([pg1, pg2, pg3], [1, 0, 2])

        r1, r2, r3 = np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])

        value11 = gaussian(c1, alpha1, R1, r1)
        value12 = gaussian(c2, alpha2, R2, r2)
        value13 = gaussian(pg3.coeff, pg3.alpha, pg3.coordinates, r3)
        v_prod_1 = value11 * value12 * value13

        value21 = gaussian(c1, alpha1, R1, r2)
        value22 = gaussian(c2, alpha2, R2, r1)
        value23 = value13
        v_prod_2 = value21 * value22 * value23

        mo = Orbital([ao1]) + Orbital([ao2])

        expected_value = v_prod_1 + v_prod_2
        actual_value = mo.get_value([r1, r2, r3])

        self.assertEqual(expected_value, actual_value)

    def test_substraction1(self):
        alpha1, alpha2 = np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        ao1 = GaussianProduct([pg1, pg2, pg3], [0, 1, 2])
        ao2 = GaussianProduct([pg1, pg2, pg3], [1, 0, 2])

        r1, r2, r3 = np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])

        value11 = gaussian(c1, alpha1, R1, r1)
        value12 = gaussian(c2, alpha2, R2, r2)
        value13 = gaussian(pg3.coeff, pg3.alpha, pg3.coordinates, r3)
        v_prod_1 = value11 * value12 * value13

        value21 = gaussian(c1, alpha1, R1, r2)
        value22 = gaussian(c2, alpha2, R2, r1)
        value23 = value13
        v_prod_2 = value21 * value22 * value23

        mo = Orbital([ao1]) - Orbital([ao2])

        expected_value = v_prod_1 - v_prod_2
        actual_value = mo.get_value([r1, r2, r3])

        self.assertEqual(expected_value, actual_value)

    def test_substraction2(self):
        alpha1, alpha2 = np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        ao1 = GaussianProduct([pg1, pg2, pg3], [0, 1, 2])
        ao2 = GaussianProduct([pg1, pg2, pg3], [1, 0, 2])
        ao3 = GaussianProduct([pg1, pg2, pg3], [1, 2, 0])
        ao4 = GaussianProduct([pg1, pg2, pg3], [2, 1, 0])

        r1, r2, r3 = np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])

        value11 = gaussian(c1, alpha1, R1, r1)
        value12 = gaussian(c2, alpha2, R2, r2)
        value13 = gaussian(pg3.coeff, pg3.alpha, pg3.coordinates, r3)
        v_prod_1 = value11 * value12 * value13

        value21 = gaussian(c1, alpha1, R1, r2)
        value22 = gaussian(c2, alpha2, R2, r1)
        value23 = value13
        v_prod_2 = value21 * value22 * value23

        value21 = gaussian(c1, alpha1, R1, r2)
        value22 = gaussian(c2, alpha2, R2, r1)
        value23 = value13
        v_prod_2 = value21 * value22 * value23

        value31 = gaussian(c1, alpha1, R1, r2)
        value32 = gaussian(c2, alpha2, R2, r3)
        value33 = gaussian(pg3.coeff, pg3.alpha, pg3.coordinates, r1)
        v_prod_3 = value31 * value32 * value33

        value41 = gaussian(c1, alpha1, R1, r3)
        value42 = gaussian(c2, alpha2, R2, r2)
        value43 = gaussian(pg3.coeff, pg3.alpha, pg3.coordinates, r1)
        v_prod_4 = value41 * value42 * value43

        mo = Orbital([ao1, ao2]) - Orbital([ao3, ao4])

        expected_value = v_prod_1 + v_prod_2 - v_prod_3 - v_prod_4
        actual_value = mo.get_value([r1, r2, r3])

        self.assertEqual(expected_value, actual_value)

    def test_multiplication1(self):
        alpha1, alpha2 = np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
        alpha3, alpha4 = np.array([0.9, 0.9, 0.9]), np.array([0.3, 0.3, 0.3])
        c1, c2, c3, c4 = 1.1, 1.5, 2.1, -1.2
        R1, R2, R3, R4 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0]), np.array([-1.1, 2.1, 0]), np.array([1.8, -2., 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg12 = pg1 * pg2
        pg3 = PrimitiveGaussian(alpha3, c3, R3)
        pg4 = PrimitiveGaussian(alpha4, c4, R4)
        pg34 = pg3 * pg4

        ao1 = GaussianProduct([pg1, pg2, pg12], [0, 1, 2])
        ao2 = GaussianProduct([pg1, pg2, pg12], [1, 0, 2])
        ao3 = GaussianProduct([pg3, pg4, pg34], [2, 1, 0])
        ao4 = GaussianProduct([pg3, pg4, pg34], [1, 2, 0])

        # mo1 = Orbital([ao1]) + Orbital([ao2])
        # mo2 = Orbital([ao3]) + Orbital([ao4])
        mo1 = Orbital([ao1, ao2])
        mo2 = Orbital([ao3, ao4])
        mo_prod = mo1 * mo2

        r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])]

        val1 = ao1.get_value(r) + ao2.get_value(r)
        val2 = ao3.get_value(r) + ao4.get_value(r)

        expected_value = np.round(val1 * val2, 15)
        actual_value = np.round(mo_prod.get_value(r), 15)

        self.assertEqual(expected_value, actual_value)

    def test_multiplication2(self):
        alpha1, alpha2 = np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
        alpha3, alpha4 = np.array([0.9, 0.9, 0.9]), np.array([0.3, 0.3, 0.3])
        c1, c2, c3, c4 = 1.1, 1.5, 2.1, -1.2
        R1, R2, R3, R4 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0]), np.array([-1.1, 2.1, 0]), np.array([1.8, -2., 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg12 = pg1 * pg2
        pg3 = PrimitiveGaussian(alpha3, c3, R3)
        pg4 = PrimitiveGaussian(alpha4, c4, R4)
        pg34 = pg3 * pg4

        ao1 = GaussianProduct([pg1, pg2, pg12], [0, 1, 2])
        ao2 = GaussianProduct([pg1, pg2, pg12], [1, 0, 2])
        ao3 = GaussianProduct([pg3, pg4, pg34], [2, 1, 0])
        ao4 = GaussianProduct([pg3, pg4, pg34], [1, 2, 0])

        mo1 = Orbital([ao1]) - Orbital([ao2])
        mo2 = Orbital([ao3]) - Orbital([ao4])
        # mo1 = Orbital([ao1, ao2])
        # mo2 = Orbital([ao3, ao4])
        mo_prod = mo1 * mo2

        r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])]

        val1 = ao1.get_value(r) - ao2.get_value(r)
        val2 = ao3.get_value(r) - ao4.get_value(r)

        expected_value = np.round(val1 * val2, 15)
        actual_value = np.round(mo_prod.get_value(r), 15)

        self.assertEqual(expected_value, actual_value)

    def test_multiplication3(self):
        alpha1, alpha2 = np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
        alpha3, alpha4 = np.array([0.9, 0.9, 0.9]), np.array([0.3, 0.3, 0.3])
        c1, c2, c3, c4 = 1.1, 1.5, 2.1, -1.2
        R1, R2, R3, R4 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0]), np.array([-1.1, 2.1, 0]), np.array([1.8, -2., 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg12 = pg1 * pg2
        pg3 = PrimitiveGaussian(alpha3, c3, R3)
        pg4 = PrimitiveGaussian(alpha4, c4, R4)
        pg34 = pg3 * pg4

        ao1 = GaussianProduct([pg1, pg2, pg12], [0, 1, 2])
        ao2 = GaussianProduct([pg1, pg2, pg12], [1, 0, 2])
        ao3 = GaussianProduct([pg3, pg4, pg34], [2, 1, 0])
        ao4 = GaussianProduct([pg3, pg4, pg34], [1, 2, 0])

        mo1 = Orbital([ao1, ao2])
        mo2 = Orbital([ao3]) - Orbital([ao4])
        # mo1 = Orbital([ao1, ao2])
        # mo2 = Orbital([ao3, ao4])
        mo_prod = mo1 * mo2

        r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])]

        val1 = ao1.get_value(r) + ao2.get_value(r)
        val2 = ao3.get_value(r) - ao4.get_value(r)

        expected_value = np.round(val1 * val2, 15)
        actual_value = np.round(mo_prod.get_value(r), 15)

        self.assertEqual(expected_value, actual_value)

    def test_normalization(self):
        sigmax = 1.0
        sigmay = 1.0
        alpha = np.array([1 / (2 * sigmax ** 2), 1 / (2 * sigmay ** 2)])
        coeff = 1

        rA = np.array([0.0, 0.0])

        pgA = PrimitiveGaussian(alpha, coeff, rA)
        pgA.coeff = pgA.coeff / np.sqrt((pgA * pgA).integrate())
        varphiA1 = Orbital([GaussianProduct([pgA], [0])])
        varphiA1.normalize()
        self.assertEqual(np.round((varphiA1**2).integrate_orbital(),10), 1.0)
    
    
if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
from src.orbitals import PrimitiveGaussian, AtomicOrbital, MolecularOrbital


def gaussian_3d(coeff, alpha, R, r):
    return coeff * np.exp(-alpha * np.dot(r - R, r - R))


class TestMolecularOrbital(unittest.TestCase):

    def test_addition(self):
        alpha1, alpha2 = 0.1, 0.5
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        ao1 = AtomicOrbital([pg1, pg2, pg3], [0, 1, 2])
        ao2 = AtomicOrbital([pg1, pg2, pg3], [1, 0, 2])

        r1, r2, r3 = np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])

        value11 = gaussian_3d(c1, alpha1, R1, r1)
        value12 = gaussian_3d(c2, alpha2, R2, r2)
        value13 = gaussian_3d(pg3.coeff, pg3.alpha, pg3.coordinates, r3)
        v_prod_1 = value11 * value12 * value13

        value21 = gaussian_3d(c1, alpha1, R1, r2)
        value22 = gaussian_3d(c2, alpha2, R2, r1)
        value23 = value13
        v_prod_2 = value21 * value22 * value23

        mo = MolecularOrbital([ao1]) + MolecularOrbital([ao2])

        expected_value = v_prod_1 + v_prod_2
        actual_value = mo.get_value([r1, r2, r3])

        self.assertEqual(expected_value, actual_value)

    def test_substraction1(self):
        alpha1, alpha2 = 0.1, 0.5
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        ao1 = AtomicOrbital([pg1, pg2, pg3], [0, 1, 2])
        ao2 = AtomicOrbital([pg1, pg2, pg3], [1, 0, 2])

        r1, r2, r3 = np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])

        value11 = gaussian_3d(c1, alpha1, R1, r1)
        value12 = gaussian_3d(c2, alpha2, R2, r2)
        value13 = gaussian_3d(pg3.coeff, pg3.alpha, pg3.coordinates, r3)
        v_prod_1 = value11 * value12 * value13

        value21 = gaussian_3d(c1, alpha1, R1, r2)
        value22 = gaussian_3d(c2, alpha2, R2, r1)
        value23 = value13
        v_prod_2 = value21 * value22 * value23

        mo = MolecularOrbital([ao1]) - MolecularOrbital([ao2])

        expected_value = v_prod_1 - v_prod_2
        actual_value = mo.get_value([r1, r2, r3])

        self.assertEqual(expected_value, actual_value)

    def test_substraction2(self):
        alpha1, alpha2 = 0.1, 0.5
        c1, c2 = 1.1, 1.5
        R1, R2 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg3 = pg1 * pg2
        ao1 = AtomicOrbital([pg1, pg2, pg3], [0, 1, 2])
        ao2 = AtomicOrbital([pg1, pg2, pg3], [1, 0, 2])
        ao3 = AtomicOrbital([pg1, pg2, pg3], [1, 2, 0])
        ao4 = AtomicOrbital([pg1, pg2, pg3], [2, 1, 0])

        r1, r2, r3 = np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])

        value11 = gaussian_3d(c1, alpha1, R1, r1)
        value12 = gaussian_3d(c2, alpha2, R2, r2)
        value13 = gaussian_3d(pg3.coeff, pg3.alpha, pg3.coordinates, r3)
        v_prod_1 = value11 * value12 * value13

        value21 = gaussian_3d(c1, alpha1, R1, r2)
        value22 = gaussian_3d(c2, alpha2, R2, r1)
        value23 = value13
        v_prod_2 = value21 * value22 * value23

        value21 = gaussian_3d(c1, alpha1, R1, r2)
        value22 = gaussian_3d(c2, alpha2, R2, r1)
        value23 = value13
        v_prod_2 = value21 * value22 * value23

        value31 = gaussian_3d(c1, alpha1, R1, r2)
        value32 = gaussian_3d(c2, alpha2, R2, r3)
        value33 = gaussian_3d(pg3.coeff, pg3.alpha, pg3.coordinates, r1)
        v_prod_3 = value31 * value32 * value33

        value41 = gaussian_3d(c1, alpha1, R1, r3)
        value42 = gaussian_3d(c2, alpha2, R2, r2)
        value43 = gaussian_3d(pg3.coeff, pg3.alpha, pg3.coordinates, r1)
        v_prod_4 = value41 * value42 * value43

        mo = MolecularOrbital([ao1, ao2]) - MolecularOrbital([ao3, ao4])

        expected_value = v_prod_1 + v_prod_2 - v_prod_3 - v_prod_4
        actual_value = mo.get_value([r1, r2, r3])

        self.assertEqual(expected_value, actual_value)

    def test_multiplication1(self):
        alpha1, alpha2, alpha3, alpha4 = 0.1, 0.5, 0.9, 0.3
        c1, c2, c3, c4 = 1.1, 1.5, 2.1, -1.2
        R1, R2, R3, R4 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0]), np.array([-1.1, 2.1, 0]), np.array([1.8, -2., 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg12 = pg1 * pg2
        pg3 = PrimitiveGaussian(alpha3, c3, R3)
        pg4 = PrimitiveGaussian(alpha4, c4, R4)
        pg34 = pg3 * pg4

        ao1 = AtomicOrbital([pg1, pg2, pg12], [0, 1, 2])
        ao2 = AtomicOrbital([pg1, pg2, pg12], [1, 0, 2])
        ao3 = AtomicOrbital([pg3, pg4, pg34], [2, 1, 0])
        ao4 = AtomicOrbital([pg3, pg4, pg34], [1, 2, 0])

        # mo1 = MolecularOrbital([ao1]) + MolecularOrbital([ao2])
        # mo2 = MolecularOrbital([ao3]) + MolecularOrbital([ao4])
        mo1 = MolecularOrbital([ao1, ao2])
        mo2 = MolecularOrbital([ao3, ao4])
        mo_prod = mo1 * mo2

        r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])]

        val1 = ao1.get_value(r) + ao2.get_value(r)
        val2 = ao3.get_value(r) + ao4.get_value(r)

        expected_value = np.round(val1 * val2, 15)
        actual_value = np.round(mo_prod.get_value(r), 15)

        self.assertEqual(expected_value, actual_value)

    def test_multiplication2(self):
        alpha1, alpha2, alpha3, alpha4 = 0.1, 0.5, 0.9, 0.3
        c1, c2, c3, c4 = 1.1, 1.5, 2.1, -1.2
        R1, R2, R3, R4 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0]), np.array([-1.1, 2.1, 0]), np.array([1.8, -2., 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg12 = pg1 * pg2
        pg3 = PrimitiveGaussian(alpha3, c3, R3)
        pg4 = PrimitiveGaussian(alpha4, c4, R4)
        pg34 = pg3 * pg4

        ao1 = AtomicOrbital([pg1, pg2, pg12], [0, 1, 2])
        ao2 = AtomicOrbital([pg1, pg2, pg12], [1, 0, 2])
        ao3 = AtomicOrbital([pg3, pg4, pg34], [2, 1, 0])
        ao4 = AtomicOrbital([pg3, pg4, pg34], [1, 2, 0])

        mo1 = MolecularOrbital([ao1]) - MolecularOrbital([ao2])
        mo2 = MolecularOrbital([ao3]) - MolecularOrbital([ao4])
        # mo1 = MolecularOrbital([ao1, ao2])
        # mo2 = MolecularOrbital([ao3, ao4])
        mo_prod = mo1 * mo2

        r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])]

        val1 = ao1.get_value(r) - ao2.get_value(r)
        val2 = ao3.get_value(r) - ao4.get_value(r)

        expected_value = np.round(val1 * val2, 15)
        actual_value = np.round(mo_prod.get_value(r), 15)

        self.assertEqual(expected_value, actual_value)

    def test_multiplication3(self):
        alpha1, alpha2, alpha3, alpha4 = 0.1, 0.5, 0.9, 0.3
        c1, c2, c3, c4 = 1.1, 1.5, 2.1, -1.2
        R1, R2, R3, R4 = np.array([0, 0, 0]), np.array([1.1, 1.1, 0]), np.array([-1.1, 2.1, 0]), np.array([1.8, -2., 0])
        pg1 = PrimitiveGaussian(alpha1, c1, R1)
        pg2 = PrimitiveGaussian(alpha2, c2, R2)
        pg12 = pg1 * pg2
        pg3 = PrimitiveGaussian(alpha3, c3, R3)
        pg4 = PrimitiveGaussian(alpha4, c4, R4)
        pg34 = pg3 * pg4

        ao1 = AtomicOrbital([pg1, pg2, pg12], [0, 1, 2])
        ao2 = AtomicOrbital([pg1, pg2, pg12], [1, 0, 2])
        ao3 = AtomicOrbital([pg3, pg4, pg34], [2, 1, 0])
        ao4 = AtomicOrbital([pg3, pg4, pg34], [1, 2, 0])

        mo1 = MolecularOrbital([ao1, ao2])
        mo2 = MolecularOrbital([ao3]) - MolecularOrbital([ao4])
        # mo1 = MolecularOrbital([ao1, ao2])
        # mo2 = MolecularOrbital([ao3, ao4])
        mo_prod = mo1 * mo2

        r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])]

        val1 = ao1.get_value(r) + ao2.get_value(r)
        val2 = ao3.get_value(r) - ao4.get_value(r)

        expected_value = np.round(val1 * val2, 15)
        actual_value = np.round(mo_prod.get_value(r), 15)

        self.assertEqual(expected_value, actual_value)

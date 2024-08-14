import unittest
import copy
import numpy as np
from src.orbitals import PrimitiveGaussian, GaussianProduct, Orbital


class TestDensities(unittest.TestCase):

    def test_integration(self):
        alpha = np.array([1.0, 1.0])
        coeff = 1

        a = 1
        b = 2
        rA = np.array([-b / 2, a / 2])
        rB = np.array([b / 2, a / 2])
        rC = np.array([b / 2, -a / 2])
        rD = np.array([-b / 2, -a / 2])

        pgA = PrimitiveGaussian(alpha, coeff, rA)
        pgB = PrimitiveGaussian(alpha, coeff, rB)
        pgC = PrimitiveGaussian(alpha, coeff, rC)
        pgD = PrimitiveGaussian(alpha, coeff, rD)

        pgA.coeff = pgA.coeff / np.sqrt((pgA * pgA).integrate())
        pgB.coeff = pgB.coeff / np.sqrt((pgB * pgB).integrate())
        pgC.coeff = pgC.coeff / np.sqrt((pgC * pgC).integrate())
        pgD.coeff = pgD.coeff / np.sqrt((pgD * pgD).integrate())

        varphiA1 = Orbital([GaussianProduct([pgA], [0])])
        varphiB1 = Orbital([GaussianProduct([pgB], [0])])
        varphiC1 = Orbital([GaussianProduct([pgC], [0])])
        varphiD1 = Orbital([GaussianProduct([pgD], [0])])

        varphiA2 = Orbital([GaussianProduct([pgA], [1])])
        varphiB2 = Orbital([GaussianProduct([pgB], [1])])
        varphiC2 = Orbital([GaussianProduct([pgC], [1])])
        varphiD2 = Orbital([GaussianProduct([pgD], [1])])

        varphiA1.normalize()
        varphiB1.normalize()
        varphiC1.normalize()
        varphiD1.normalize()

        varphiA2.normalize()
        varphiB2.normalize()
        varphiC2.normalize()
        varphiD2.normalize()

        phi_g1 = copy.deepcopy(varphiA1) + copy.deepcopy(varphiB1) + copy.deepcopy(varphiC1) + copy.deepcopy(varphiD1)
        phi_e1 = copy.deepcopy(varphiA1) - copy.deepcopy(varphiB1) - copy.deepcopy(varphiC1) + copy.deepcopy(varphiD1)

        phi_g2 = copy.deepcopy(varphiA2) + copy.deepcopy(varphiB2) + copy.deepcopy(varphiC2) + copy.deepcopy(varphiD2)
        phi_e2 = copy.deepcopy(varphiA2) - copy.deepcopy(varphiB2) - copy.deepcopy(varphiC2) + copy.deepcopy(varphiD2)

        phi_g1.normalize()
        phi_e1.normalize()

        phi_g2.normalize()
        phi_e2.normalize()

        rho1 = 1 / 2 * (phi_g1 ** 2) + 1 / 2 * (phi_e1 ** 2)
        rho2 = 1 / 6 * (phi_g1 ** 2) * (phi_g2 ** 2) + 1 / 6 * (phi_e1 ** 2) * (phi_e2 ** 2) + 1 / 3 * (phi_g1 ** 2) * (
                    phi_e2 ** 2) + 1 / 3 * (phi_e1 ** 2) * (phi_g2 ** 2) - 1 / 3 * phi_g1 * phi_e1 * phi_g2 * phi_e2

        rho2_from1 = rho2.integrate(1)

        point1 = np.array([0.5, 0.2])
        point2 = np.array([0.93, -0.14])
        point3 = np.array([1.8, 0.4])

        expected1 = np.round(rho1.get_value([point1]), 12)
        actual1 = np.round(rho2_from1.get_value([point1]), 12)
        self.assertEqual(expected1, actual1)

        expected2 = np.round(rho1.get_value([point2]), 12)
        actual2 = np.round(rho2_from1.get_value([point2]), 12)
        self.assertEqual(expected2, actual2)

        expected3 = np.round(rho1.get_value([point3]), 12)
        actual3 = np.round(rho2_from1.get_value([point3]), 12)
        self.assertEqual(expected3, actual3)
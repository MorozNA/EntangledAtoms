import unittest
import numpy as np
from src.orbitals import PrimitiveGaussian, AtomicOrbital, MolecularOrbital
from src.orbitals import YoungDiagram


class TestYoundDiagram(unittest.TestCase):

    def test_as_3(self):
        diagram = [[1, 2], [0]]
        YD = YoungDiagram([diagram])

        YD.symm_col(0)
        YD.symm_col(1)
        YD.antisymm_row(0)
        YD.antisymm_row(1)

        alpha = 0.9
        coeff = 1.1

        a = 0.5
        triangle_radius = np.array([0.0, a * np.sqrt(3) / 6, 0.0])
        R1 = np.array([a / 2, 0.0, 0.0]) - triangle_radius
        R2 = np.array([0.0, a / 2 * np.sqrt(3), 0.0]) - triangle_radius
        R3 = np.array([-a / 2, 0.0, 0.0]) - triangle_radius

        pg1 = PrimitiveGaussian(alpha, coeff, R1)
        pg2 = PrimitiveGaussian(alpha, coeff, R2)
        pg3 = PrimitiveGaussian(alpha, coeff, R3)

        mo = YD.get_orbital([pg1, pg2, pg3])

        ao1 = AtomicOrbital([pg1, pg2, pg3], [1, 2, 0])
        ao2 = AtomicOrbital([pg1, pg2, pg3], [0, 2, 1])
        ao3 = AtomicOrbital([pg1, pg2, pg3], [2, 1, 0])
        ao4 = AtomicOrbital([pg1, pg2, pg3], [2, 0, 1])  # TODO: here draft of article contains mistake

        mo1 = MolecularOrbital([ao1, ao2])
        mo2 = MolecularOrbital([ao3, ao4])
        analytical_mo = mo1 - mo2

        r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])]

        self.assertEqual(mo.get_value(r), analytical_mo.get_value(r))

    def test_sa_3(self):
        diagram = [[1, 2], [0]]
        YD = YoungDiagram([diagram])

        YD.antisymm_col(0)
        YD.antisymm_col(1)
        YD.symm_row(0)
        YD.symm_row(1)

        alpha = 0.9
        coeff = 1.1

        a = 0.5
        triangle_radius = np.array([0.0, a * np.sqrt(3) / 6, 0.0])
        R1 = np.array([a / 2, 0.0, 0.0]) - triangle_radius
        R2 = np.array([0.0, a / 2 * np.sqrt(3), 0.0]) - triangle_radius
        R3 = np.array([-a / 2, 0.0, 0.0]) - triangle_radius

        pg1 = PrimitiveGaussian(alpha, coeff, R1)
        pg2 = PrimitiveGaussian(alpha, coeff, R2)
        pg3 = PrimitiveGaussian(alpha, coeff, R3)

        mol_orb = YD.get_orbital([pg1, pg2, pg3])

        ao1 = AtomicOrbital([pg1, pg2, pg3], [1, 2, 0])
        ao2 = AtomicOrbital([pg1, pg2, pg3], [0, 2, 1])
        ao3 = AtomicOrbital([pg1, pg2, pg3], [2, 1, 0])
        ao4 = AtomicOrbital([pg1, pg2, pg3], [2, 0, 1])  # TODO: here draft of article contains mistake

        mo1 = MolecularOrbital([ao1])
        mo2 = MolecularOrbital([ao2])
        mo3 = MolecularOrbital([ao3])
        mo4 = MolecularOrbital([ao4])
        analytical_mo = mo1 - mo2 + mo3 - mo4

        r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65])]
        self.assertEqual(mol_orb.get_value(r), analytical_mo.get_value(r))

    def test_as_4_step3(self):
        diagram = [[0, 1], [2, 3]]
        YD = YoungDiagram([diagram])

        YD.symm_col(0)
        YD.symm_col(1)
        YD.antisymm_row(0)
        # YD.antisymm_row(1)

        alpha = 0.9
        coeff = 1.1

        a = 0.5
        R1 = np.array([-a / 2, -a / 2, 0.0])
        R2 = np.array([-a / 2, a / 2, 0.0])
        R3 = np.array([a / 2, a / 2, 0.0])
        R4 = np.array([a / 2, -a / 2, 0.0])

        pg1 = PrimitiveGaussian(alpha, coeff, R1)
        pg2 = PrimitiveGaussian(alpha, coeff, R2)
        pg3 = PrimitiveGaussian(alpha, coeff, R3)
        pg4 = PrimitiveGaussian(alpha, coeff, R4)
        pg_list = [pg1, pg2, pg3, pg4]

        mol_orb = YD.get_orbital(pg_list)

        mo11 = MolecularOrbital([AtomicOrbital([pg1, pg3], [0, 2]), AtomicOrbital([pg1, pg3], [2, 0])])
        mo12 = MolecularOrbital([AtomicOrbital([pg2, pg4], [1, 3]), AtomicOrbital([pg2, pg4], [3, 1])])
        mo1 = mo11 * mo12

        ao21 = AtomicOrbital([pg1, pg2, pg3, pg4], [1, 0, 2, 3])
        ao22 = AtomicOrbital([pg1, pg2, pg3, pg4], [1, 2, 0, 3])
        ao23 = AtomicOrbital([pg1, pg2, pg3, pg4], [3, 0, 2, 1])
        ao24 = AtomicOrbital([pg1, pg2, pg3, pg4], [3, 2, 0, 1])
        mo2 = MolecularOrbital([ao21, ao22, ao23, ao24])

        analytical_mo = mo1 - mo2
        # mo2.ao_list

        r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65]),
             np.array([-0.39, 0.15, 0.37])]

        self.assertEqual(mol_orb.get_value(r), analytical_mo.get_value(r))



    def test_sa_4_step3(self):
        diagram = [[0, 1], [2, 3]]
        YD = YoungDiagram([diagram])

        YD.antisymm_col(0)
        YD.antisymm_col(1)
        YD.symm_row(0)
        # YD.antisymm_row(1)

        alpha = 0.9
        coeff = 1.1

        a = 0.5
        R1 = np.array([-a / 2, -a / 2, 0.0])
        R2 = np.array([-a / 2, a / 2, 0.0])
        R3 = np.array([a / 2, a / 2, 0.0])
        R4 = np.array([a / 2, -a / 2, 0.0])

        pg1 = PrimitiveGaussian(alpha, coeff, R1)
        pg2 = PrimitiveGaussian(alpha, coeff, R2)
        pg3 = PrimitiveGaussian(alpha, coeff, R3)
        pg4 = PrimitiveGaussian(alpha, coeff, R4)
        pg_list = [pg1, pg2, pg3, pg4]

        mol_orb = YD.get_orbital(pg_list)

        mo11 = MolecularOrbital([AtomicOrbital([pg1, pg3], [0, 2])]) - MolecularOrbital([AtomicOrbital([pg1, pg3], [2, 0])])
        mo12 = MolecularOrbital([AtomicOrbital([pg2, pg4], [1, 3])]) - MolecularOrbital([AtomicOrbital([pg2, pg4], [3, 1])])
        mo1 = mo11 * mo12

        mo21 = MolecularOrbital([AtomicOrbital([pg1, pg4], [1, 3])]) - MolecularOrbital([AtomicOrbital([pg1, pg4], [3, 1])])
        mo22 = MolecularOrbital([AtomicOrbital([pg2, pg3], [0, 2])]) - MolecularOrbital([AtomicOrbital([pg2, pg3], [2, 0])])
        mo2 = mo21 * mo22

        analytical_mo = mo1 + mo2

        r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65]),
             np.array([-0.39, 0.15, 0.37])]

        expected_val = np.round(mol_orb.get_value(r), 15)
        actual_val = np.round(analytical_mo.get_value(r), 15)

        self.assertEqual(expected_val, actual_val)
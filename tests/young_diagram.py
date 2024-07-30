import numpy as np
from src.orbitals import PrimitiveGaussian, AtomicOrbital, MolecularOrbital
from src.orbitals import YoungDiagram

diagram = [[0, 1], [2, 3]]
YD = YoungDiagram([diagram])

YD.symm_col(0)
YD.symm_col(1)
YD.antisymm_row(0)
print(YD)

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
for ao in analytical_mo.ao_list:
    if ao.get_coeff()>0:
        print('+')
    elif ao.get_coeff()<0:
        print('-')
    else:
        print('0')

r = [np.array([0.15, 0.15, 0.15]), np.array([-0.15, -0.15, -0.15]), np.array([0.45, 0.55, 0.65]),
     np.array([-0.39, 0.15, 0.37])]

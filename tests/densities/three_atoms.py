import copy

import numpy as np
from src.orbitals import PrimitiveGaussian, GaussianProduct, Orbital

del_x2 = 1 / 2

alpha = np.array([1 / 2, 1 / 2])
coeff = 1 / (np.pi ** (1/4))

a = 2.0 * del_x2
h = 2.5 * del_x2
rA = np.array([-h / 2, 0])
rB = np.array([h / 2, a / 2])
rC = np.array([h / 2, -a / 2])

x1_min = -5 * del_x2
x1_max = 5 * del_x2
x2_min = -5 * del_x2
x2_max = 5 * del_x2
step = 0.025

x1_1, x1_2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))


pgA = PrimitiveGaussian(alpha, coeff, rA)
pgB = PrimitiveGaussian(alpha, coeff, rB)
pgC = PrimitiveGaussian(alpha, coeff, rC)

pgA.coeff = pgA.coeff / np.sqrt((pgA * pgA).integrate())
pgB.coeff = pgA.coeff / np.sqrt((pgB * pgB).integrate())
pgC.coeff = pgA.coeff / np.sqrt((pgC * pgC).integrate())

pgA_val = pgA.get_values(x1_1, x1_2)
pgB_val = pgB.get_values(x1_1, x1_2)
pgC_val = pgC.get_values(x1_1, x1_2)

# TODO: find a way to normalize simpler
varphiA1 = Orbital([GaussianProduct([pgA], [0])])
varphiB1 = Orbital([GaussianProduct([pgB], [0])])
varphiC1 = Orbital([GaussianProduct([pgC], [0])])

Sab = (pgA * pgB).integrate()
Sbc = (pgB * pgC).integrate()
Sac = (pgA * pgC).integrate()

q = 1 / np.sqrt(3 + 2 * Sab + 2 * Sbc + 2 * Sac)
p = 1 / np.sqrt(2 * (1 - Sbc))
f = (2 * (1 + Sbc) + Sab + Sac) / (1 + Sab + Sac)

phi_g1_orb = q * (copy.deepcopy(varphiA1) + copy.deepcopy(varphiB1) + copy.deepcopy(varphiC1))
phi_e1_orb = q * f * copy.deepcopy(varphiA1) - q * (copy.deepcopy(varphiB1) + copy.deepcopy(varphiC1))
module_g1 = phi_g1_orb * phi_g1_orb.conj()
module_e1 = phi_e1_orb * phi_e1_orb.conj()
K_g1 = module_g1.integrate_orbital()
K_e1 = module_e1.integrate_orbital()

phi_g1 = q * (pgA_val + pgB_val + pgC_val)
phi_e1 = q * f * pgA_val - q * (pgB_val + pgC_val)

phi_g1 /= np.sqrt(K_g1)
phi_e1 /= np.sqrt(K_e1)

el_point1 =[0.0, 0.0]
x2_1, x2_2 = el_point1[0], el_point1[1]
point1 = np.array([x2_1, x2_2], dtype=float)
phi_g2 = q * (pgA.get_value(point1) + pgB.get_value(point1) + pgC.get_value(point1))
phi_e2 = q * f * pgA.get_value(point1) - q * (pgB.get_value(point1) + pgC.get_value(point1))
prob1 = 2 / 3 * (phi_g2 ** 2) + 1 / 3 * (phi_e2 ** 2)
rho2_1 = 1/3 * ((phi_g1 ** 2) * (phi_g2 ** 2)) + 1/3 * ((phi_g1 ** 2) * (phi_e2 ** 2)) \
       + 1/3 * ((phi_e1 ** 2) * (phi_g2 ** 2)) - 1/3 * (phi_g1 * phi_e1 * phi_g2 * phi_e2)
rho2_1 = rho2_1 / prob1

el_point2 =[h / 2, 0]
x2_1, x2_2 = el_point2[0], el_point2[1]
point2 = np.array([x2_1, x2_2], dtype=float)
phi_g2 = q * (pgA.get_value(point2) + pgB.get_value(point2) + pgC.get_value(point2))
phi_e2 = q * f * pgA.get_value(point2) - q * (pgB.get_value(point2) + pgC.get_value(point2))
prob2 = 2 / 3 * (phi_g2 ** 2) + 1 / 3 * (phi_e2 ** 2)
rho2_2 = 1/3 * ((phi_g1 ** 2) * (phi_g2 ** 2)) + 1/3 * ((phi_g1 ** 2) * (phi_e2 ** 2)) \
       + 1/3 * ((phi_e1 ** 2) * (phi_g2 ** 2)) - 1/3 * (phi_g1 * phi_e1 * phi_g2 * phi_e2)
rho2_2 = rho2_2 / prob1

el_point3 =[-h / 2, 0]
x2_1, x2_2 = el_point3[0], el_point3[1]
point3 = np.array([x2_1, x2_2], dtype=float)
phi_g2 = q * (pgA.get_value(point3) + pgB.get_value(point3) + pgC.get_value(point3))
phi_e2 = q * f * pgA.get_value(point3) - q * (pgB.get_value(point3) + pgC.get_value(point3))
prob3 = 2 / 3 * (phi_g2 ** 2) + 1 / 3 * (phi_e2 ** 2)
rho2_3 = 1/3 * ((phi_g1 ** 2) * (phi_g2 ** 2)) + 1/3 * ((phi_g1 ** 2) * (phi_e2 ** 2)) \
       + 1/3 * ((phi_e1 ** 2) * (phi_g2 ** 2)) - 1/3 * (phi_g1 * phi_e1 * phi_g2 * phi_e2)
rho2_3 = rho2_3 / prob3


rho1 = 2 / 3 * (phi_g1 ** 2) + 1 / 3 * (phi_e1 ** 2)
#3 rho1_sq = rho1 ** 2
# rho2_same = 1/6 * (phi_g1 ** 4) + 1/6 * (phi_e1 ** 4) + 1/3 * ((phi_g1 ** 2) * (phi_e1 ** 2))
# rho2 = 1/6 * ((phi_g1 ** 2) * (phi_g2 ** 2)) + 1/6 * ((phi_e1 ** 2) * (phi_e2 ** 2)) \
#        + 1/3 * ((phi_g1 ** 2) * (phi_e2 ** 2)) + 1/3 * ((phi_e1 ** 2) * (phi_g2 ** 2)) \
#        -1/3 * (phi_g1 * phi_e1 * phi_g2 * phi_e2)
# rho2 = rho2 / prob2

# rho_plus = 1 / 2 * (phi_g1 ** 2) + 1 / 2 * (phi_plus ** 2)
# rho_im = 1 / 2 * (phi_g1 ** 2) + 1 / 2 * (phi_im * phi_im.conj())
# rho_im = np.real(rho_im)
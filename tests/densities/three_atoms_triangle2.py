import copy

import numpy as np
from src.orbitals import PrimitiveGaussian, GaussianProduct, Orbital

del_x2 = 1 / 2

alpha = np.array([1 / 2, 1 / 2])
coeff = 1 / (np.pi ** (1/4))

a = 4.0 * del_x2
h = np.sqrt(3) / 2 * a
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

phi_g1_orb = q * (copy.deepcopy(varphiA1) + copy.deepcopy(varphiB1) + copy.deepcopy(varphiC1))
phi_e1_orb = q * f * copy.deepcopy(varphiA1) - q * (copy.deepcopy(varphiB1) + copy.deepcopy(varphiC1))
phi_ee1_orb = p * copy.deepcopy(varphiB1) - p * copy.deepcopy(varphiC1)
module_g1 = phi_g1_orb * phi_g1_orb.conj()
module_e1 = phi_e1_orb * phi_e1_orb.conj()
module_ee1 = phi_ee1_orb * phi_ee1_orb.conj()
K_g1 = module_g1.integrate_orbital()
K_e1 = module_e1.integrate_orbital()
K_ee1 = module_ee1.integrate_orbital()

phi_g1 = q * (pgA_val + pgB_val + pgC_val)
phi_e1 = q * f * pgA_val - q * (pgB_val + pgC_val)
phi_ee1 = p * (pgB_val - pgC_val)

phi_g1 /= np.sqrt(K_g1)
phi_e1 /= np.sqrt(K_e1)
phi_ee1 /= np.sqrt(K_ee1)

phi_plus = (phi_e1 + phi_ee1) / np.sqrt(2)
phi_minus = (phi_e1 - phi_ee1) / np.sqrt(2)
phi_im = (phi_e1 + 1j * phi_ee1) / np.sqrt(2)

rho_im2 = 1 / 2 * (phi_g1 ** 2) + 1 / 2 * (phi_im * phi_im.conj())
rho_im2 = np.real(rho_im2)
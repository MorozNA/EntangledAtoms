import copy

import numpy as np
from src.orbitals import PrimitiveGaussian, GaussianProduct, Orbital

sigmax = 1.0
sigmay = 1.0
alpha = np.array([1 / (2 * sigmax**2), 1 / (2 * sigmay**2)])
coeff = 1.

a = 2.0 * sigmax
b = np.sqrt(3) / 2 * a
# h = 1.5 * a
# tr = np.array([0.0, a * np.sqrt(3) / 6]) # triangle radius
shift = np.array([b / 2, 0.0])
rA = np.array([0.0, 0.0]) - shift
rB = np.array([b, a / 2]) - shift
rC = np.array([b, -a / 2]) - shift

pgA = PrimitiveGaussian(alpha, coeff, rA)
pgB = PrimitiveGaussian(alpha, coeff, rB)
pgC = PrimitiveGaussian(alpha, coeff, rC)

pgA.coeff = pgA.coeff / np.sqrt((pgA * pgA).integrate())
pgB.coeff = pgA.coeff / np.sqrt((pgB * pgB).integrate())
pgC.coeff = pgA.coeff / np.sqrt((pgC * pgC).integrate())

varphiA1 = Orbital([GaussianProduct([pgA], [0])])
varphiB1 = Orbital([GaussianProduct([pgB], [0])])
varphiC1 = Orbital([GaussianProduct([pgC], [0])])

varphiA2 = Orbital([GaussianProduct([pgA], [1])])
varphiB2 = Orbital([GaussianProduct([pgB], [1])])
varphiC2 = Orbital([GaussianProduct([pgC], [1])])

varphiA1.normalize()
varphiB1.normalize()
varphiC1.normalize()

varphiA2.normalize()
varphiB2.normalize()
varphiC2.normalize()

Sab = (pgA * pgB).integrate()
Sbc = (pgB * pgC).integrate()
Sac = (pgA * pgC).integrate()

q = 1 / np.sqrt(3 + 2 * Sab + 2 * Sbc + 2 * Sac)
p = 1 / np.sqrt(2 * (1 - Sbc))
f = (2 * (1 + Sbc) + Sab + Sac) / (1 + Sab + Sac)

phi_g1 = q * (copy.deepcopy(varphiA1) + copy.deepcopy(varphiB1) + copy.deepcopy(varphiC1))
phi_e1 = q * f * copy.deepcopy(varphiA1) - q * (copy.deepcopy(varphiB1) + copy.deepcopy(varphiC1))

phi_g2 = q * (copy.deepcopy(varphiA2) + copy.deepcopy(varphiB2) + copy.deepcopy(varphiC2))
phi_e2 = q * f * copy.deepcopy(varphiA2) - q * (copy.deepcopy(varphiB2) + copy.deepcopy(varphiC2))

phi_g1.normalize()
phi_e1.normalize()

phi_g2.normalize()
phi_e2.normalize()

rho1 = 2/3 * phi_g1**2 + 1/3 * phi_e1**2
rho1_sq = rho1 ** 2
rho2_same = 1/3 * (phi_g1 ** 4) + 1/3 * ((phi_g1 ** 2) * (phi_e1 ** 2))
rho2 = 1/3 * ((phi_g1 ** 2) * (phi_g2 ** 2)) + 1/3 * ((phi_g1 ** 2) * (phi_e2 ** 2)) \
       + 1/3 * ((phi_e1 ** 2) * (phi_g2 ** 2)) - 1/3 * (phi_g1 * phi_e1 * phi_g2 * phi_e2)

print(rho1.integrate_orbital())
print(rho1_sq.integrate_orbital())
print(rho2_same.integrate_orbital())
print('\n')

print(len(rho1.ao_list))
print(len(rho1_sq.ao_list))
print(len(rho2_same.ao_list))

from plot_tool import get_rho1_vec

r_atomic = [rA, rB, rC]
side = np.linalg.norm(r_atomic[0] - r_atomic[1])
x1_min = -1.75 * side
x1_max = 1.75 * side
x2_min = -1.75 * side
x2_max = 1.75 * side
step = 0.025

x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
y1 = get_rho1_vec(x1, x2, rho1)
y2 = get_rho1_vec(x1, x2, rho1_sq)
y3 = get_rho1_vec(x1, x2, rho2_same)

np.savetxt('3 atoms/data/x1_mesh.txt', x1, fmt='%f')
np.savetxt('3 atoms/data/x2_mesh.txt', x2, fmt='%f')
np.savetxt('3 atoms/data/rho1.txt', y1, fmt='%f')
np.savetxt('3 atoms/data/rho1_sq.txt', y2, fmt='%f')
np.savetxt('3 atoms/data/rho2_same.txt', y3, fmt='%f')

from plot_tool import get_rho2_vec
point_x = (rA[0] + rB[0]) / 2
point_y = rA[1]
y = get_rho2_vec(point_x, point_y, x1, x2, rho2)
np.savetxt('3 atoms/data/rho2.txt', y, fmt='%f')

# from plot_tool import plot_rho1
# r_atomic = [rA, rB, rC]
# plot_rho1(rho1, r_atomic, 'fig3')
#
# title2 = r'Electron Density Function (Squared) $[\rho(\mathbf{r})]^2$'
# plot_rho1(rho1_sq, r_atomic, 'fig3_sq', title2)
# title3 = r'Coincidence Point Density Function  $\rho(\mathbf{r}, \mathbf{r})$'
# plot_rho1(rho2_same, r_atomic, 'fig3_point', title3)
#
# rho_diff = rho1_sq - rho2_same
# title_diff = r'Difference  $[\rho(\mathbf{r})]^2 - \rho(\mathbf{r}, \mathbf{r})$'
# plot_rho1(rho_diff, r_atomic, 'fig3_diff', title_diff)
#
# # TODO: Абсолютные значения непонятны
# # TODO: + - сбивает с толка
# # TODO: vmin не всегда помогает бороться с черным фоном

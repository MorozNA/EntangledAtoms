import copy

import numpy as np
from src.orbitals import PrimitiveGaussian, GaussianProduct, Orbital

sigmax = 1.0
sigmay = 1.0
alpha = np.array([1 / (2 * sigmax**2), 1 / (2 * sigmay**2)])
coeff = 1

b = 2 * sigmax
a = 1 * sigmay
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
rho2_integrated = rho2.integrate(1)

print(rho1.integrate_orbital())
print(rho2.integrate_orbital())
print('\n')

print(len(rho1.ao_list))
print(len(rho2.ao_list))
print(len(rho2_integrated.ao_list))

#
#
#
from plot_tool import plot_rho1, plot_rho2
plot_rho1(rho1, rA, rB, rC, rD)
plot_rho2(rho2, -b/2, 0.0, '-\\frac{b}{2}', '0', rA, rB, rC, rD)
import numpy as np
from functions import pg


sigmax = 1.0
sigmay = 1.0
alpha = np.array([1 / (2 * sigmax**2), 1 / (2 * sigmay**2)])
coeff = 1.

b = 2.0 * sigmax
a = 2.0 * sigmay
rA = np.array([-b / 2, a / 2])
rB = np.array([b / 2, a / 2])
rC = np.array([b / 2, -a / 2])
rD = np.array([-b / 2, -a / 2])
side = a

x1_min = -1.75 * side
x1_max = 1.75 * side
x2_min = -1.75 * side
x2_max = 1.75 * side
step = 0.025

x1_1, x1_2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))

pgA = pg(coeff, alpha, rA, x1_1, x1_2)
pgB = pg(coeff, alpha, rB, x1_1, x1_2)
pgC = pg(coeff, alpha, rC, x1_1, x1_2)
pgD = pg(coeff, alpha, rD, x1_1, x1_2)

phi_g1 = pgA + pgB + pgC + pgD
phi_e1 = pgA - pgB - pgC + pgD

rho1 = 1 / 2 * (phi_g1 ** 2) + 1 / 2 * (phi_e1 ** 2)
rho1_sq = rho1 ** 2
rho2_same = 1/6 * (phi_g1 ** 4) + 1/6 * (phi_e1 ** 4) + 1/3 * ((phi_g1 ** 2) * (phi_e1 ** 2))

import seaborn as sns
colormap = sns.color_palette("Reds", as_cmap=True)
from tests.densities.plot_tool import plot_rho1

x1 = x1_1 / a
x2 = x1_2 / a
rA, rB, rC, rD = rA / a, rB/ a, rC/ a, rD/ a

r_atomic = [rA, rB, rC, rD]

plot_rho1(rho1, r_atomic, 'fig1_analytical', y=rho1, colormap=colormap)
plot_rho1(rho1_sq, r_atomic, 'fig2_analytical', y=rho1_sq, colormap=colormap)
plot_rho1(rho2_same, r_atomic, 'fig3_analytical', y=rho2_same, colormap=colormap)
import numpy as np

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

import seaborn as sns
# TODO: find better color palette
colormap = sns.color_palette("Reds", as_cmap=True)

from plot_tool import plot_rho1
from plot_tool import plot_rho1
x1 = np.loadtxt('./3 atoms/data/x1_mesh.txt', dtype=float)
x2 = np.loadtxt('./3 atoms/data/x2_mesh.txt', dtype=float)
rho1 = np.loadtxt('./3 atoms/data/rho1.txt', dtype=float)
rho1_sq = np.loadtxt('./3 atoms/data/rho1_sq.txt', dtype=float)
rho2_same = np.loadtxt('./3 atoms/data/rho2_same.txt', dtype=float)
rho2 = np.loadtxt('./3 atoms/data/rho2.txt', dtype=float)

# Scaling
x1 = x1 / a
x2 = x2 / a
rA, rB, rC = rA / a, rB/ a, rC/ a
r_atomic = [rA, rB, rC]

plot_rho1(rho1, r_atomic, 'fig3', y=rho1, colormap=colormap)

title2 = r'Electron Density Function (Squared) $[\rho(\mathbf{r})]^2$'
plot_rho1(rho1_sq, r_atomic, 'fig3_sq', title2, y=rho1_sq, colormap=colormap)

title3 = r'Coincidence Point Density Function  $\rho(\mathbf{r}, \mathbf{r})$'
plot_rho1(rho2_same, r_atomic, 'fig3_point', title3, y=rho2_same, colormap=colormap)

title4 = r'Conditional Density Function $\rho(\mathbf{r}_1 \mid \mathbf{r}_2 = (0, 0))$'
point_x = (rA[0] + rB[0]) / 2
point_y = rA[1]
el_point =[point_x, point_y]
plot_rho1(rho2, r_atomic, 'fig3_case', title4, y=rho2, colormap=colormap, el_point=el_point)
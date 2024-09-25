import numpy as np

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

import seaborn as sns
colormap = sns.color_palette("Reds", as_cmap=True)

from plot_tool import plot_rho1
x1 = np.loadtxt('./4 atoms/data/x1_mesh.txt', dtype=float)
x2 = np.loadtxt('./4 atoms/data/x2_mesh.txt', dtype=float)
rho1 = np.loadtxt('./4 atoms/data/rho1.txt', dtype=float)
rho1_sq = np.loadtxt('./4 atoms/data/rho1_sq.txt', dtype=float)
rho2_same = np.loadtxt('./4 atoms/data/rho2_same.txt', dtype=float)
rho2_left = np.loadtxt('./4 atoms/data/rho2.txt', dtype=float)
rho2_right = np.loadtxt('./4 atoms/data/rho2_right.txt', dtype=float)

# Scaling
x1 = x1 / a
x2 = x2 / a
rA, rB, rC, rD = rA / a, rB/ a, rC/ a, rD/ a

r_atomic = [rA, rB, rC, rD]

plot_rho1(rho1, r_atomic, 'fig4', y=rho1, colormap=colormap)

title2 = r'Electron Density Function (Squared) $[\rho(\mathbf{r})]^2$'
plot_rho1(rho1_sq, r_atomic, 'fig4_sq', title2, y=rho1_sq, colormap=colormap)

title3 = r'Coincidence Point Density Function  $\rho(\mathbf{r}, \mathbf{r})$'
plot_rho1(rho2_same, r_atomic, 'fig4_point', title3, y=rho2_same, colormap=colormap)

title4 = r'Conditional Density Function $\rho(\mathbf{r}_1 \mid \mathbf{r}_2 = (a/2, 0))$'
el_point =[rB[0], (rB[1] + rC[1]) / 2]
plot_rho1(rho2_right, r_atomic, 'fig4_case', title4, y=rho2_right, colormap=colormap, el_point=el_point)
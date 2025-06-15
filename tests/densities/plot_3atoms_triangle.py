import numpy as np
from three_atoms_triangle1 import rA, rB, rC, del_x2
from three_atoms_triangle1 import x1_1, x1_2
from three_atoms_triangle1 import rho1, rho_plus_tri, rho_minus_tri, rho_im1_tri
from three_atoms_triangle2 import rho_im2
from plot_tool import plot_four
import seaborn as sns

colormap = sns.color_palette("OrRd", as_cmap=True)
# colormap = sns.color_palette("Spectral_r", as_cmap=True)
# colormap = sns.color_palette("YlOrRd", as_cmap=True)
# colormap = sns.light_palette("darkorange", as_cmap=True)

# Scaling
x1_1 = x1_1 / del_x2
x1_2 = x1_2 / del_x2
rA, rB, rC = rA / del_x2, rB/ del_x2, rC/ del_x2


r_atomic = [rA, rB, rC]

path = './3 atoms/pictures/'


title = r'different basises'
rho_list = [[rho_plus_tri, rho_minus_tri], [rho_im1_tri, rho_im2]]
plot_four(rho_list, r_atomic, name='fig_basises', colormap=colormap, path=path)

import numpy as np
from four_atoms_square1 import rA, rB, rC, rD, del_x2
from four_atoms_square1 import x1_1, x1_2
from four_atoms_square1 import rho_plus, rho_im1
from three_atoms_triangle1 import rho_plus_tri, rho_im1_tri
from plot_tool import plot_four
import seaborn as sns

colormap = sns.color_palette("OrRd", as_cmap=True)
# colormap = sns.color_palette("Spectral_r", as_cmap=True)
# colormap = sns.color_palette("YlOrRd", as_cmap=True)
# colormap = sns.light_palette("darkorange", as_cmap=True)

# Scaling
x1_1 = x1_1 / del_x2
x1_2 = x1_2 / del_x2
rA, rB, rC, rD = rA / del_x2, rB/ del_x2, rC/ del_x2, rD/ del_x2


r_atomic = [rA, rB, rC, rD]

path = './4 atoms/pictures/'


title = r'3and4'
rho_list = [[rho_plus, rho_im1], [rho_plus_tri, rho_im1_tri]]
plot_four(rho_list, r_atomic, name='3and4', colormap=colormap, path=path)

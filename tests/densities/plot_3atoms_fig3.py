import numpy as np
from three_atoms import rA, rB, rC, del_x2
from three_atoms import x1_1, x1_2, h
from three_atoms import rho1, rho2_1, rho2_2, rho2_3
from plot_tool import plot_rho1, plot_two, plot_four
import seaborn as sns

colormap = sns.color_palette("OrRd", as_cmap=True)
# colormap = sns.color_palette("Spectral_r", as_cmap=True)

# colormap = sns.color_palette("RdYlBu_r", as_cmap=True)
# colormap = sns.color_palette("RdGy_r", as_cmap=True)
# colormap = sns.color_palette("RdBu_r", as_cmap=True)

# colormap = sns.color_palette("nipy_spectral", as_cmap=True)


# Scaling
x1_1 = x1_1 / del_x2
x1_2 = x1_2 / del_x2
rA, rB, rC = rA / del_x2, rB / del_x2, rC / del_x2
h = h / del_x2


r_atomic = [rA, rB, rC]

path = './3 atoms/pictures/'

title1 = r'Probability Distribution  $\rho(\mathbf{r})$'
plot_rho1(rho1, r_atomic, name='fig3_v2', title=title1, y=rho1, colormap=colormap, path=path)


title_two = r'Probability Distribution'
rho_list = [[rho1, rho2_1], [rho2_3, rho2_2]]
el_point1 =[0.0, 0.0]
el_point2 =[h / 2, 0.0]
el_point3 =[-h / 2, 0.0]
el_point_list = [el_point1, el_point3, el_point2]
plot_four(rho_list, r_atomic, el_points=el_point_list, name='fig_double', colormap=colormap, path=path)


# import pandas as pd
# data_fig3a = [x1_1, x1_2, rho1]
# data_fig3b = [x1_1, x1_2, rho2_1]
# data_fig3c = [x1_1, x1_2, rho2_3]
# data_fig3d = [x1_1, x1_2, rho2_2]
#
# def save_to_excel(data, filename):
#     """Helper function to save a list of arrays to an Excel file"""
#     # Convert each array to a DataFrame
#     df_x1 = pd.DataFrame(data[0])
#     df_x2 = pd.DataFrame(data[1])
#     df_rho = pd.DataFrame(data[2])
#
#     # Create an Excel writer object
#     with pd.ExcelWriter(filename) as writer:
#         df_x1.to_excel(writer, sheet_name='x1', index=False)
#         df_x2.to_excel(writer, sheet_name='x2', index=False)
#         df_rho.to_excel(writer, sheet_name='rho', index=False)
#
#
# # Save each dataset to separate files
# save_to_excel(data_fig3a, '3 atoms/data/fig3/figure_3a_data.xlsx')
# save_to_excel(data_fig3b, '3 atoms/data/fig3/figure_3b_data.xlsx')
# save_to_excel(data_fig3c, '3 atoms/data/fig3/figure_3c_data.xlsx')
# save_to_excel(data_fig3d, '3 atoms/data/fig3/figure_3d_data.xlsx')


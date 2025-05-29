import numpy as np
from four_atoms import rA, rB, rC, rD, del_x2
from four_atoms import x1_1, x1_2, x2_1, x2_2
from four_atoms import rho1, rho2_1, rho2_2, rho2_3
from plot_tool import plot_rho1, plot_four
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

title1 = r'Probability Distribution  $\rho(\mathbf{r})$'
plot_rho1(rho1, r_atomic, name='fig4_v2', title=title1, y=rho1, colormap=colormap, path=path)

# title2 = r'$\rho(\mathbf{r})_+$'
# plot_rho1(rho_plus, r_atomic, name='fig4_plus', title=title2, y=rho_plus, colormap=colormap, path=path)
#
# title3 = r'$\rho(\mathbf{r})_i$'
# plot_rho1(rho_im, r_atomic, name='fig4_im', title=title3, y=rho_im, colormap=colormap, path=path)

# title4 = r'Conditional Probability Distribution $\rho(\mathbf{r}_1 \mid \mathbf{r}_2 = (a/2, 0))$'
# el_point =[x2_1, x2_2]
# plot_rho1(rho2, r_atomic, 'fig4_case_v2', title4, y=rho2, colormap=colormap, el_point=el_point, path=path)


title_two = r'Probability Distribution'
rho_list = [[rho1, rho2_2], [rho2_1, rho2_3]]
el_point1 =[rB[0], (rB[1] + rC[1]) / 2]
el_point2 =[0, 0]
el_point3 =[rB[0], rB[1]]
el_point_list = [el_point2, el_point1, el_point3]
plot_four(rho_list, r_atomic, el_points=el_point_list, name='fig_double', colormap=colormap, path=path)


import pandas as pd
data_fig4a = [x1_1, x1_2, rho1]
data_fig4b = [x1_1, x1_2, rho2_1]
data_fig4c = [x1_1, x1_2, rho2_3]
data_fig4d = [x1_1, x1_2, rho2_2]

def save_to_excel(data, filename):
    """Helper function to save a list of arrays to an Excel file"""
    # Convert each array to a DataFrame
    df_x1 = pd.DataFrame(data[0])
    df_x2 = pd.DataFrame(data[1])
    df_rho = pd.DataFrame(data[2])

    # Create an Excel writer object
    with pd.ExcelWriter(filename) as writer:
        df_x1.to_excel(writer, sheet_name='x1', index=False)
        df_x2.to_excel(writer, sheet_name='x2', index=False)
        df_rho.to_excel(writer, sheet_name='rho', index=False)


# Save each dataset to separate files
save_to_excel(data_fig4a, '4 atoms/data/fig4/figure_4a_data.xlsx')
save_to_excel(data_fig4b, '4 atoms/data/fig4/figure_4b_data.xlsx')
save_to_excel(data_fig4c, '4 atoms/data/fig4/figure_4c_data.xlsx')
save_to_excel(data_fig4d, '4 atoms/data/fig4/figure_4d_data.xlsx')
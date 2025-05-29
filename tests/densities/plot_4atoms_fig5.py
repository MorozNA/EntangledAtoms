import numpy as np
from four_atoms_square1 import rA, rB, rC, rD, del_x2
from four_atoms_square1 import x1_1, x1_2
from four_atoms_square1 import rho1, rho2, rho_plus, rho_minus, rho_im1
# from four_atoms_square2 import rho_im2
from plot_tool import plot_four, plot_six
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


title = r'different basises'
# rho_list = [[rho_plus, rho_minus], [rho_im1, rho_im2]]
rho_six = [[rho1, rho2], [rho_plus, rho_minus], [rho_im1, rho_im1]]
# plot_four(rho_list, r_atomic, name='fig_basises4', colormap=colormap, path=path)
plot_six(rho_six, r_atomic, name='fig_basises6', colormap=colormap, path=path)



import pandas as pd
data_fig5a_1 = [x1_1, x1_2, rho1]
data_fig5a_2 = [x1_1, x1_2, rho2]
data_fig5b_1 = [x1_1, x1_2, rho_plus]
data_fig5b_2 = [x1_1, x1_2, rho_minus]
data_fig5c_1 = [x1_1, x1_2, rho_im1]
data_fig5c_2 = [x1_1, x1_2, rho_im1]

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
save_to_excel(data_fig5a_1, '4 atoms/data/fig5/figure_5a_1_data.xlsx')
save_to_excel(data_fig5a_2, '4 atoms/data/fig5/figure_5a_2_data.xlsx')
save_to_excel(data_fig5b_1, '4 atoms/data/fig5/figure_5b_1_data.xlsx')
save_to_excel(data_fig5b_2, '4 atoms/data/fig5/figure_5b_2_data.xlsx')
save_to_excel(data_fig5c_1, '4 atoms/data/fig5/figure_5c_1_data.xlsx')
save_to_excel(data_fig5c_2, '4 atoms/data/fig5/figure_5c_2_data.xlsx')
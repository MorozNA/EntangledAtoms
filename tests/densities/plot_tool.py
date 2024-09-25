import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from pylab import figure

plt.rcParams["font.family"] = "Times New Roman"


def get_rho1(x1, x2, rho):
    coordinates = np.array([x1, x2])
    return rho.get_value([coordinates])


def get_rho2(x1, x2, y1, y2, rho):
    coordinates = np.array([x1, x2])
    return rho.get_value([coordinates, np.array([y1, y2])])


get_rho1_vec = np.vectorize(get_rho1, excluded='rho')
get_rho2_vec = np.vectorize(get_rho2, excluded='rho y1 y2')

import seaborn as sns


def plot_rho1(rho1, r_atomic, name='fig', title=None, colormap='mako', y=None, el_point=None):
    if title is None:
        title = r'Electron Density Function  $\rho(x, y)$'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.major.width'] = 4
    plt.rcParams['ytick.major.width'] = 4
    plt.rcParams['xtick.major.size'] = 15
    plt.rcParams['ytick.major.size'] = 15
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['axes.linewidth'] = 4

    side = np.linalg.norm(r_atomic[0] - r_atomic[1])
    x1_min = -1.75 * side
    x1_max = 1.75 * side
    x2_min = -1.75 * side
    x2_max = 1.75 * side
    step = 0.025

    if y is None:
        x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
        y = get_rho1_vec(x1, x2, rho1)

    fig, ax = plt.subplots(figsize=(16, 16.5))
    plt.subplots_adjust(top=0.82)
    plt.subplots_adjust(right=0.905)
    plt.subplots_adjust(left=-0.1)
    plt.subplots_adjust(bottom=0.15)
    # TODO: Adjust vmin
    # colormap = "ch:start=.2,rot=-.3"
    # colormap = "light:b"
    # colormap = "ch:s=.25,rot=-.25"
    # colormap = "PuBu"
    # sns.color_palette("blend:#7AB,#EDA", as_cmap=True)
    # cmap = sns.color_palette(colormap, as_cmap=True)
    im = ax.imshow(y, extent=[x1_min, x1_max, x2_min, x2_max], cmap=colormap, origin='lower', aspect='equal')
    # im = ax.contourf(x2, x1, y, cmap=sns.color_palette("mako", as_cmap=True), levels=30, vmin=-0.01)
    cbar = plt.colorbar(im)
    cbar.set_ticks([])
    cbar.set_label(label='-          Electron probability distribution          +', size=40, labelpad=20)
    plt.rcParams['text.color'] = '#000000'
    title_color = '#000000'
    ax.tick_params(axis='x', colors='#000000')
    ax.tick_params(axis='y', colors='#000000')
    ax.set_title(title, fontsize=44, loc='left', color=title_color, y=1.02)
    # ax.text(-2.9, 2.6, (r'$\rho(x,y)$'), color='#dfdfdf', fontsize=30)

    plt.xlabel(r'$x/a$', fontsize=40)
    plt.ylabel(r'$y/a$', fontsize=40)
    # plt.yticks(range(-2, 3, 1))

    plt.scatter(r_atomic[0][0], r_atomic[0][1], s=800, c='black', marker='o', label=r'Atomic position')
    for i in range(1, len(r_atomic)):
        plt.scatter(r_atomic[i][0], r_atomic[i][1], s=800, c='black', marker='o')

    if el_point is None:
        pass
    else:
        plt.axhline(y=el_point[1], linestyle='--', dashes=(5,5), color='black')
        plt.axvline(x=el_point[0], linestyle='--', dashes=(5,5), color='black')
        # plt.scatter(el_point[0], el_point[1], s=100, c='black', marker='o')

    plt.legend(loc='lower left', fontsize=40)
    plt.savefig((name + '.png'), bbox_inches='tight')
    # plt.show()

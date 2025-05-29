import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from pylab import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['axes.linewidth'] = 2


def get_rho1(x1, x2, rho):
    coordinates = np.array([x1, x2])
    return rho.get_value([coordinates])


def get_rho2(x1, x2, y1, y2, rho):
    coordinates = np.array([x1, x2])
    return rho.get_value([coordinates, np.array([y1, y2])])


get_rho1_vec = np.vectorize(get_rho1, excluded='rho')
get_rho2_vec = np.vectorize(get_rho2, excluded='rho y1 y2')


def plot_rho1(rho1, r_atomic=None, name='fig', title=None, colormap='mako', y=None, el_point=None, path='./'):
    if title is None:
        title = r'Electron Density Function  $\rho(x, y)$'

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
    im = ax.imshow(y, extent=[x1_min, x1_max, x2_min, x2_max], cmap=colormap, origin='lower', aspect='equal')
    # im = ax.contourf(x2, x1, y, cmap=sns.color_palette("mako", as_cmap=True), levels=30, vmin=-0.01)
    cbar = plt.colorbar(im)
    # cbar.set_ticks([])
    # cbar.set_label(label='-          Electron probability distribution          +', size=40, labelpad=20)
    plt.rcParams['text.color'] = '#000000'
    title_color = '#000000'
    ax.tick_params(axis='x', colors='#000000')
    ax.tick_params(axis='y', colors='#000000')
    ax.set_title(title, fontsize=44, loc='left', color=title_color, y=1.02)
    # ax.text(-2.9, 2.6, (r'$\rho(x,y)$'), color='#dfdfdf', fontsize=30)

    plt.xlabel(r'$x/a$', fontsize=40)
    plt.ylabel(r'$y/a$', fontsize=40)
    # plt.yticks(range(-2, 3, 1))

    # if r_atomic is None:
    #     pass
    # else:
    #     plt.scatter(r_atomic[0][0], r_atomic[0][1], s=800, c='black', marker='o', label=r'Atomic position')
    #     for i in range(1, len(r_atomic)):
    #         plt.scatter(r_atomic[i][0], r_atomic[i][1], s=800, c='black', marker='o')

    if el_point is None:
        pass
    else:
        plt.axhline(y=el_point[1], linestyle='--', dashes=(5, 5), color='black')
        plt.axvline(x=el_point[0], linestyle='--', dashes=(5, 5), color='black')
        # plt.scatter(el_point[0], el_point[1], s=100, c='black', marker='o')

    plt.legend(loc='lower left', fontsize=40)
    plt.savefig((path + name + '.png'), bbox_inches='tight')


def plot_two(rho1, rho2, r_atomic, name='fig', title=None, colormap='mako', el_point=None, path='./',
             el_point_text='(a/2, 0)'):
    side = np.linalg.norm(r_atomic[0] - r_atomic[1])
    x1_min = -1.75 * side
    x1_max = 1.75 * side
    x2_min = -1.75 * side
    x2_max = 1.75 * side

    fig, axes = plt.subplots(figsize=(20, 8.5), nrows=1, ncols=2)
    # fig.suptitle("Probability Distributions", fontsize=32)

    vmax = np.max([np.max(rho2), np.max(rho1)])

    rho = [rho1, rho2]
    for i in range(2):
        im = axes[i].imshow(rho[i], extent=[x1_min, x1_max, x2_min, x2_max], vmin=0, vmax=vmax, cmap=colormap,
                            aspect="auto")
        axes[i].set_xticks(np.arange(-1.5, 2.0, 0.5))
        axes[i].set_yticks(np.arange(-1.5, 2.0, 0.5))
        axes[i].set_xlabel(r'$x/a$', fontsize=32)
        axes[i].set_ylabel(r'$y/a$', fontsize=32)
        axes[i].grid(color='gray', linestyle='-', linewidth=0.25)

    # axes[0].set_title(r'$\rho(\mathbf{r})$', fontsize=32, pad=15)
    # axes[1].set_title(r'$\rho(\mathbf{r} \mid \mathbf{r}_2 = ' + el_point_text + ')$', fontsize=32, pad=15)

    if el_point is None:
        pass
    else:
        axes[1].scatter(el_point[0], el_point[1], marker='+', s=500, linewidth=3.5, color='black')
        axes[1].axhline(y=el_point[1], linestyle='--', dashes=(8, 8), linewidth=1.0, color='black')
        axes[1].axvline(x=el_point[0], linestyle='--', dashes=(8, 8), linewidth=1.0, color='black')

    fig.tight_layout(pad=5.0)
    # fig.subplots_adjust(right=0.8)
    # divider = make_axes_locatable(axes[1])
    # cax = divider.append_axes('right', size='5%', pad=0.5)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig((path + name + '.png'), bbox_inches='tight')


def plot_four(rho_list, r_atomic, name='fig', colormap='mako', el_points=None, path='./', el_point_text='(a/2, 0)'):
    del_x2 = np.linalg.norm(r_atomic[0] - r_atomic[1]) / 2.5
    x1_min = -5.25 * del_x2
    x1_max = 5.25 * del_x2
    x2_min = -5.25 * del_x2
    x2_max = 5.25 * del_x2

    fig, axes = plt.subplots(figsize=(9, 7.53), nrows=2, ncols=2, sharex=True, sharey=True)
    # fig.suptitle("Probability Distribution", fontsize=32)

    index_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i in index_list:
        axes[i[0]][i[1]].set_xticks(np.arange(-6.0, 6.0, 2.0))
        axes[i[0]][i[1]].set_yticks(np.arange(-6.0, 6.0, 2.0))

        axes[i[0]][i[1]].set_xticks(np.arange(-6.0, 6.0, 1.0), minor=True)
        axes[i[0]][i[1]].set_yticks(np.arange(-6.0, 6.0, 1.0), minor=True)

        # axes[i[0]][i[1]].set_axisbelow(True)
        axes[i[0]][i[1]].grid(color='black', linestyle='-', linewidth=1.2, which='major', alpha=1.0, zorder=-1.0)
        axes[i[0]][i[1]].grid(color='black', linestyle='-', linewidth=1.2, which='minor', alpha=1.0, zorder=-1.0)

        # axes[i[0]][i[1]].xaxis.set_ticks_position('none')
        # axes[i[0]][i[1]].yaxis.set_ticks_position('none')

        # axes[i[0]][i[1]].set_xlabel(r'$x/a$', fontsize=32)
        # axes[i[0]][i[1]].set_ylabel(r'$y/a$', fontsize=32)
        axes[i[0]][i[1]].set_aspect('equal')  # adjustable='box',

    vmax = np.max(rho_list)  # - 0.01 # - 0.01 used for better visualization (first two pictures)
    for i in range(len(rho_list)):
        for j in range(len(rho_list[i])):
            im = axes[i][j].imshow(rho_list[i][j], extent=[x1_min, x1_max, x2_min, x2_max], vmin=0.0, vmax=vmax, cmap=colormap, zorder=2.0, alpha=0.90)  # alpha = 0.97, zorder = 2.0
            # for atom in r_atomic:
            #     axes[i][j].scatter(atom[0], atom[1])

    # for i in index_list:
    #     axes[i[0]][i[1]].xaxis.set_ticks_position('none')
    #     axes[i[0]][i[1]].yaxis.set_ticks_position('none')

    fig.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)

    index_list2 = [[0, 1], [1, 0], [1, 1]]
    if el_points is None:
        pass
    else:
        for i in range(len(el_points)):
            axes[index_list2[i][0]][index_list2[i][1]].scatter(el_points[i][0], el_points[i][1], marker='+', s=500,
                                                             linewidth=3.5, color='black', zorder=3.0)
            axes[index_list2[i][0]][index_list2[i][1]].axhline(y=el_points[i][1], linestyle='--', dashes=(8, 8),
                                                             linewidth=1.0, color='black', zorder=3.0)
            axes[index_list2[i][0]][index_list2[i][1]].axvline(x=el_points[i][0], linestyle='--', dashes=(8, 8),
                                                             linewidth=1.0, color='black', zorder=3.0)

    # fig.subplots_adjust(right=0.8)
    # divider = make_axes_locatable(axes[1])
    # cax = divider.append_axes('right', size='5%', pad=0.5)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    axes[0][0].tick_params(which='both', axis='x', length=0)
    axes[0][1].tick_params(which='both', axis='both', length=0)
    # axes[1][0].tick_params(which='both', axis='x', length=0)
    axes[1][1].tick_params(which='both', axis='y', length=0)


    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fig.colorbar(im, ax=axes.ravel().tolist(), ticks=[0.00, 0.05, 0.10, 0.15, 0.20])
    plt.savefig((path + name + '.png'), bbox_inches='tight')


def plot_six(rho_list, r_atomic, name='fig', colormap='mako', path='./'):
    del_x2 = np.linalg.norm(r_atomic[0] - r_atomic[1]) / 2.5
    x1_min = -5.25 * del_x2
    x1_max = 5.25 * del_x2
    x2_min = -5.25 * del_x2
    x2_max = 5.25 * del_x2

    fig, axes = plt.subplots(figsize=(9, 11), nrows=3, ncols=2, sharex='col', sharey='row')
    # fig.suptitle("Probability Distribution", fontsize=32)

    index_list = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    for i in index_list:
        axes[i[0]][i[1]].set_xticks(np.arange(-6.0, 6.0, 2.0))
        axes[i[0]][i[1]].set_yticks(np.arange(-6.0, 6.0, 2.0))

        axes[i[0]][i[1]].set_xticks(np.arange(-6.0, 6.0, 1.0), minor=True)
        axes[i[0]][i[1]].set_yticks(np.arange(-6.0, 6.0, 1.0), minor=True)

        # axes[i[0]][i[1]].set_axisbelow(True)
        axes[i[0]][i[1]].grid(color='black', linestyle='-', linewidth=1.2, which='major', alpha=1.0, zorder=-1.0)
        axes[i[0]][i[1]].grid(color='black', linestyle='-', linewidth=1.2, which='minor', alpha=1.0, zorder=-1.0)

        # axes[i[0]][i[1]].xaxis.set_ticks_position('none')
        # axes[i[0]][i[1]].yaxis.set_ticks_position('none')

        # axes[i[0]][i[1]].set_xlabel(r'$x/a$', fontsize=32)
        # axes[i[0]][i[1]].set_ylabel(r'$y/a$', fontsize=32)
        axes[i[0]][i[1]].set_aspect('equal')  # adjustable='box',

    vmax = np.max(rho_list)  # - 0.01 # - 0.01 used for better visualization (first two pictures)
    for i in range(len(rho_list)):
        for j in range(len(rho_list[i])):
            im = axes[i][j].imshow(rho_list[i][j], extent=[x1_min, x1_max, x2_min, x2_max], vmin=0.0, vmax=vmax, cmap=colormap, zorder=2.0, alpha=0.90)  # alpha = 0.97, zorder = 2.0
            # for atom in r_atomic:
            #     axes[i][j].scatter(atom[0], atom[1])

    # for i in index_list:
    #     axes[i[0]][i[1]].xaxis.set_ticks_position('none')
    #     axes[i[0]][i[1]].yaxis.set_ticks_position('none')

    fig.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)

    # fig.subplots_adjust(right=0.8)
    # divider = make_axes_locatable(axes[1])
    # cax = divider.append_axes('right', size='5%', pad=0.5)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    axes[0][0].tick_params(which='both', axis='x', length=0)
    axes[0][1].tick_params(which='both', axis='both', length=0)
    # axes[1][0].tick_params(which='both', axis='x', length=0)
    axes[1][1].tick_params(which='both', axis='y', length=0)
    axes[2][1].tick_params(which='both', axis='y', length=0)


    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fig.colorbar(im, ax=axes.ravel().tolist(), ticks=[0.00, 0.05, 0.10, 0.15, 0.20])
    plt.savefig((path + name + '.png'), bbox_inches='tight')

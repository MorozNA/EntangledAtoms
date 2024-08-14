import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, cm

def get_rho1(x1, x2, rho):
    coordinates = np.array([x1, x2])
    return rho.get_value([coordinates])


def get_rho2(x1, x2, y1, y2, rho):
    coordinates = np.array([x1, x2])
    return rho.get_value([coordinates, np.array([y1, y2])])


get_rho1_vec = np.vectorize(get_rho1, excluded='rho')
get_rho2_vec = np.vectorize(get_rho2, excluded='rho y1 y2')


def plot_rho1(rho1, rA, rB, rC, rD):

    x1_min = -3.1
    x1_max = 3.1
    x2_min = -3.1
    x2_max = 3.1
    step = 0.05

    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    y = get_rho1_vec(x1, x2, rho1)
    # y = plot_rho1_vec(x1, x2, rho2_integrated)

    plt.imshow(y, extent=[x1_min, x1_max, x2_min, x2_max], cmap=cm.jet, origin='lower', aspect='equal')
    plt.colorbar()
    plt.contour(x1, x2, y, extent=[x1_min, x1_max, x2_min, x2_max], colors='black', origin='lower', linewidths=0.7)

    plt.title(r'$\rho(x, y)$', fontsize=20)
    plt.xlabel(r'$x/a$', fontsize=16)
    plt.ylabel(r'$y/a$', fontsize=16)
    plt.yticks(range(-2, 3, 1))

    plt.scatter(rA[0], rA[1], s=40, c='black', marker='o', label=r'Atomic position')
    plt.scatter(rB[0], rB[1], s=40, c='black', marker='o')
    plt.scatter(rC[0], rC[1], s=40, c='black', marker='o')
    plt.scatter(rD[0], rD[1], s=40, c='black', marker='o')
    #plt.savefig("evaluate_2d_function_using_meshgrid_03.png", bbox_inches='tight')
    plt.legend(loc='lower left')
    plt.show()


def plot_rho2(rho2, y1, y2, y1_str, y2_str, rA, rB, rC, rD):
    x1_min = -3.1
    x1_max = 3.1
    x2_min = -3.1
    x2_max = 3.1
    step = 0.05

    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))

    y = get_rho2_vec(x1, x2, y1, y2, rho2)

    plt.imshow(y, extent=[x1_min, x1_max, x2_min, x2_max], cmap=cm.jet, origin='lower')
    plt.colorbar()
    plt.contour(x1, x2, y, extent=[x1_min, x1_max, x2_min, x2_max], colors='black', origin='lower', linewidths=0.7)

    plt.title(r'$\rho(x_1, y_1; x_2=' + y1_str + ', y_2=' + y2_str + ')$', fontsize=20)
    plt.xlabel(r'$x_1/a$', fontsize=16)
    plt.ylabel(r'$y_1/a$', fontsize=16)
    plt.yticks(range(-2, 4, 1))

    plt.scatter(rA[0], rA[1], s=40, c='black', marker='o', label=r'Atomic position')
    plt.scatter(rB[0], rB[1], s=40, c='black', marker='o')
    plt.scatter(rC[0], rC[1], s=40, c='black', marker='o')
    plt.scatter(rD[0], rD[1], s=40, c='black', marker='o')
    #plt.savefig("evaluate_2d_function_using_meshgrid_03.png", bbox_inches='tight')
    plt.legend(loc='lower left')
    # plt.savefig("evaluate_2d_function_using_meshgrid_03.png", bbox_inches='tight')
    plt.show()
